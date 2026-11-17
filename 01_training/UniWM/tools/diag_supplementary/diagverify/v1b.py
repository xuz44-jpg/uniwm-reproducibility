from v_common import *
R=load_var()
for comp,ci in (('dx',0),('dy',1),('dyaw',2)):
    S=per_state(R,ci)
    w=np.array([s['w'] for s in S]); braw=np.array([s['braw'] for s in S])
    order=np.argsort(-(w+braw))[:6]
    print('==',comp)
    for i in order:
        s=S[i]
        print(f" {s['traj']} k={s['k']} n={s['n']} rec_k={np.round(s['rec'],3)} rec_next={np.round(s['recn'],3) if s['recn'] else None} w={s['w']:.5f} b={s['b']:.5f}")
        print('   imag frames:',[list(np.round(f,2)) for f in s['frames']])
        print('   real:',list(np.round(s['refs'],2)),' vq:',list(np.round(s['refqs'],2)))
