"""Frozen-checkpoint native inference; no optimizer and no future-reference inputs."""
import gc
import math
from pathlib import Path
import random
import sys
from types import SimpleNamespace
import numpy as np
import torch
from PIL import Image
from suite_common import digest


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def pil(path, resize=None):
    with Image.open(path) as im:
        im = im.convert('RGB')
        return im.resize(resize) if resize else im.copy()


class ReConBackend:
    def __init__(self, manifest, root):
        import yaml
        self.name = manifest['model']
        repo = root / 'repos' / self.name
        sys.path.insert(0, str(repo))
        import misc
        from models import CDiT_models
        self.misc = misc
        cfg_path = root / f'configs/{self.name}_a30.yaml'
        cfg = yaml.safe_load(cfg_path.read_text())
        data_cfg = yaml.safe_load((repo/'config/data_config.yaml').read_text())
        self.spacing = float(data_cfg['recon']['metric_waypoint_spacing'])
        self.stats = {k:np.asarray(v,dtype=float)[None] for k,v in data_cfg['action_stats'].items()}
        self.config_sha256 = digest(cfg_path)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        if self.name == 'nwm':
            from diffusion import create_diffusion
            from diffusers.models import AutoencoderKL
            self.model = CDiT_models[cfg['model']](context_size=4,input_size=28,in_channels=4)
            self.codec = AutoencoderKL.from_pretrained(cfg['vae_path'],local_files_only=True).cuda().eval()
            self.sampler = create_diffusion('250')
        else:
            sys.path.insert(0,str(repo/'RAE/src'))
            from RAE.src.utils.model_utils import instantiate_from_config
            from RAE.src.utils.train_utils import parse_configs
            from RAE.src.stage2.transport.transport import Transport,ModelType,PathType,WeightType,Sampler
            c,*_ = parse_configs(cfg['config_path'])
            self.codec = instantiate_from_config(c).cuda().eval()
            self.model = CDiT_models[cfg['model']](context_size=4,input_size=16,in_channels=self.codec.latent_dim,
                learn_sigma=False,head_width=2048,head_depth=2,head_num_heads=16)
            transport = Transport(model_type=ModelType.VELOCITY,path_type=PathType.LINEAR,
                loss_type=WeightType.VELOCITY,time_dist_type='uniform',
                time_dist_shift=math.sqrt(self.codec.latent_dim*16*16/4096),train_eps=1e-3,sample_eps=1e-3)
            self.sampler = Sampler(transport).sample_ode(sampling_method='euler',num_steps=50,atol=1e-6,rtol=1e-3,reverse=False)
        weights = torch.load(manifest['checkpoint'],map_location='cpu',weights_only=False)['ema']
        print('STRICT_LOAD', self.model.load_state_dict({k.removeprefix('_orig_mod.'):v for k,v in weights.items()},strict=True),flush=True)
        del weights; gc.collect()
        self.model.cuda().eval()

    def encode_action(self, action):
        y = np.asarray(action,dtype=float).copy()
        y[:2] = self.misc.normalize_data(y[None,:2]/self.spacing,self.stats)[0]
        return y

    def predict(self, case, resolve):
        hashes = {}
        def frame(path):
            p = resolve(path); hashes[str(p)] = digest(p)
            if path.startswith('@'):
                feedback = p.with_suffix('.pt'); hashes[str(feedback)] = digest(feedback)
                x = torch.load(feedback,map_location='cuda',weights_only=True)
                assert torch.isfinite(x).all()
                return x
            return self.misc.transform(pil(p)).cuda()[None]
        # Free RAE history consists of native latents; NWM retains float pixels.
        # RNG reset before encoding also controls NWM's stochastic VAE posterior.
        seed_all(case['seed'])
        with torch.amp.autocast('cuda',dtype=torch.bfloat16):
            xs = [frame(p) for p in case['history_paths']]
            assert len(xs)==4
            if self.name=='raenwm' and case['history']=='free':
                # Batch the remaining observed frames exactly as at the first
                # teacher step; generated latents bypass the encoder entirely.
                real=[i for i,p in enumerate(case['history_paths']) if not p.startswith('@')]
                if real:
                    enc=self.codec.encode(torch.cat([xs[i] for i in real],dim=0)*.5+.5)
                    for j,i in enumerate(real):xs[i]=enc[j:j+1]
                cond=torch.stack(xs,dim=1)
            elif self.name=='raenwm':
                x=torch.stack(xs,dim=1)
                cond = self.codec.encode(x.flatten(0,1)*.5+.5).unflatten(0,(1,4))
            else:
                x=torch.stack(xs,dim=1)
                cond = self.codec.encode(x.flatten(0,1)).latent_dist.sample().mul_(.18215).unflatten(0,(1,4))
            y = torch.tensor(self.encode_action(case['action']),device='cuda',dtype=torch.float32)[None]
            t = torch.tensor([case['horizon_frames']/128],device='cuda',dtype=torch.float32)
            observed=[]
            def hook(_,inputs,kw):
                assert torch.equal(kw['y'],y) and torch.equal(kw['rel_t'],t)
                if not observed: observed.append(kw['y'].detach().float().cpu().tolist())
            handle=self.model.register_forward_pre_hook(hook,with_kwargs=True)
            try:
                if self.name=='nwm':
                    noise=torch.randn(1,4,28,28,device='cuda')
                    latent=self.sampler.p_sample_loop(self.model,noise.shape,noise,clip_denoised=False,
                        model_kwargs=dict(y=y,x_cond=cond,rel_t=t),progress=False,device='cuda')
                    pixels=self.codec.decode(latent/.18215).sample
                    assert torch.isfinite(pixels).all()
                    # Native rollout clips floating-point decoded pixels before re-encoding.
                    feedback=pixels.clamp(-1,1)
                    display=feedback*.5+.5
                else:
                    noise=torch.randn(1,self.codec.latent_dim,16,16,device='cuda')
                    latent=self.sampler(noise,self.model,y=y,x_cond=cond,rel_t=t)[-1]
                    pixels=self.codec.decode(latent).float()
                    assert torch.isfinite(pixels).all()
                    feedback=latent
                    display=pixels.clamp(0,1)
            finally: handle.remove()
        assert observed and torch.isfinite(latent).all()
        array=(display[0].float().permute(1,2,0).cpu().numpy()*255).round().astype('uint8')
        if self.name=='raenwm' and case['history']=='free':
            with torch.amp.autocast('cuda',dtype=torch.bfloat16):
                anchor=self.codec.decode(xs[-1]).float().clamp(0,1)
        else:anchor=(xs[-1]*.5+.5).clamp(0,1)
        anchor_image=Image.fromarray((anchor[0].float().permute(1,2,0).cpu().numpy()*255).round().astype('uint8'))
        return Image.fromarray(array),feedback.detach().cpu(),dict(observed_network_y=observed[0],rel_t=float(t[0]),input_hashes=hashes,_anchor=anchor_image)


class UniWMBackend:
    def __init__(self, manifest, root):
        repo=Path('external/uniwm-action-pilot/repo')
        sys.path.insert(0,str(repo))
        from scripts.load_model import load_model
        from scripts.action_utils import action_to_text
        from scripts.prompt_builder import build_viz_prompt
        from uniwm.pose_memory import MemoryEncoder, retrieve, relative_pose, memory_block, splice_memory, MEM_PLACEHOLDER
        self.action_text,self.prompt=action_to_text,build_viz_prompt
        self.retrieve,self.relative_pose,self.block,self.splice,self.marker=retrieve,relative_pose,memory_block,splice_memory,MEM_PLACEHOLDER
        torch.backends.cuda.matmul.allow_tf32=False
        torch.backends.cudnn.allow_tf32=False
        seed_all(17)
        args=SimpleNamespace(model='anole',model_ckpt=manifest['checkpoint'],image_seq_length=784,precision='fp16',
            attn_impl='sdpa',base_model='external/uniwm-action-pilot/models/anole',use_memory_bank_inference=False,
            do_task_level_eval=False,do_train=False,do_single_step_eval=False,do_rollout_eval=False,
            do_action_probe=True,lora_r=16,lora_alpha=32,lora_dropout=.05,lora_targets='q_proj,k_proj,v_proj,o_proj',
            lm_head_mode='full',fp32_mlp_layers='auto',init_adapter=None,lora_fp32=True)
        mp=load_model(args,None); self.model,self.processor=mp['model'],mp['processor'];self.model.eval()
        from verify_uniwm_embeddings import verify_loaded_embeddings
        self.audit=verify_loaded_embeddings(self.model,self.processor.tokenizer,Path(manifest['checkpoint']))
        self.encoder=MemoryEncoder(self.model,self.processor,res=160)
        self.cache={}
        self.sampling=dict(num_beams=1,do_sample=True,max_new_tokens=804,temperature=.7,top_p=.9)

    def decode(self,tokens):
        pixels=self.model.decode_image_tokens(tokens)
        array=self.processor.postprocess_pixel_values(pixels)[0].cpu().numpy().transpose(1,2,0)
        assert np.isfinite(array).all()
        return Image.fromarray(array.astype('uint8'))

    def sanitize(self,path,generated):
        key=(str(path),generated)
        if key not in self.cache:
            image=pil(path,None if generated else (256,256))
            inputs=self.processor(text='<image>',images=image,return_tensors='pt').to(self.model.device)
            tokens=self.model.model.model.get_image_tokens(inputs['pixel_values'].to(self.model.dtype))
            self.cache[key]=self.decode(tokens)
        return self.cache[key]

    def predict(self,case,resolve):
        hashes={}
        def load(path,sanitize=False):
            p=resolve(path);hashes[str(p)]=digest(p)
            return self.sanitize(p,path.startswith('@')) if sanitize else pil(p,(256,256))
        current=load(case['history_paths'][-1],sanitize=True)
        start,goal=load(case['start_image']),load(case['goal_image'])
        text=self.action_text(case['action']); assert text==case['action_text']
        sp=case['start_pose']; pose=f'Starting Point Coordinate: x={sp[0]:.3f}, y={sp[1]:.3f}, yaw={sp[2]:.3f}\n'
        entries=case['memory_entries']
        assert all(e['index']<case['current_index'] for e in entries)
        if case['memory_mode']!='none':
            selected=self.retrieve(case['query_pose'],[e['pose'] for e in entries],2,mode=case['memory_mode'])
            rel=[self.relative_pose(case['query_pose'],entries[i]['pose']) for i in selected]
            images=[load(entries[i]['path'],sanitize=entries[i]['path'].startswith('@')) for i in selected]
            mem_tokens=self.encoder.encode(images); block=self.block(rel,text='pose')
        else: selected=[];mem_tokens=[];block=''
        prompt=self.prompt(text,pose,memory_block=block)
        inputs=self.processor(text=[prompt],images=[start,goal,current],return_tensors='pt',padding=False,truncation=False).to(self.model.device)
        if mem_tokens:
            inputs['input_ids']=self.splice(inputs['input_ids'],mem_tokens,self.processor.tokenizer.convert_tokens_to_ids(self.marker),self.model.config.boi_token_id,self.model.config.eoi_token_id)
            inputs['attention_mask']=torch.ones_like(inputs['input_ids'])
        ids=inputs['input_ids'][0].tolist()
        import re
        action_ids=[self.processor.tokenizer.convert_tokens_to_ids(x) for x in re.findall(r'<[^>]+>',text)]
        assert len(action_ids)==3 and all(x in ids for x in action_ids)
        assert len(ids)+804<=4096, 'Prompt and output exceed context budget'
        seed_all(case['seed'])
        with torch.autocast(device_type='cuda',dtype=torch.float16):
            row=self.model.generate(**inputs,**self.sampling)[0]
        if row.ndim==2: row=row[0]
        begin=(row==self.model.config.boi_token_id).nonzero().flatten()
        if begin.numel()!=1: raise ValueError('Expected one generated image segment')
        b=int(begin[0]);end=b+785
        if len(row)<=end or int(row[end])!=self.model.config.eoi_token_id: raise ValueError('Incomplete generated image')
        image=self.decode(row[b+1:end].unsqueeze(0))
        info=dict(action_token_ids=action_ids,prompt_sha256=__import__('hashlib').sha256(prompt.encode()).hexdigest(),
            input_ids_sha256=__import__('hashlib').sha256(np.asarray(ids,dtype=np.int64).tobytes()).hexdigest(),
            input_tokens=len(ids),memory_indices=[entries[i]['index'] for i in selected],
            memory_token_count=sum(len(t)+2 for t in mem_tokens),input_hashes=hashes,_anchor=current)
        return image,None,info
