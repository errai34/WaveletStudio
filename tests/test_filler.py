import numpy as np

def cookie_cutter(img, stx, sty, M, N):
    piece = img[stx : min(stx+M, img.shape[0]), sty : min(sty+N, img.shape[1])]
    px = piece.shape[0]
    py = piece.shape[1]
    res = np.zeros((M, N))
    for m in range(M // piece.shape[0] + (0 if M % piece.shape[0] == 0 else 1)):
        for n in range(N // piece.shape[1] + (0 if N % piece.shape[1] == 0 else 1)):
            print('m', m)
            print('n')
            filler = piece if m % 2 == 0 else np.flip(piece, 0)
            filler = filler if n % 2 == 0 else np.flip(filler, 1)
            #filler = piece

            rlimx = min((m+1)*px, M) - m*px
            rlimy = min((n+1)*py, N) - n*py
            res[m*px:rlimx+m*px, n*py:rlimy+n*py] = filler[0:rlimx, 0:rlimy]
    return res

x = 3
y = 5
img = np.arange(0, x*y, 1).reshape(x, y)
print(img)

print(cookie_cutter(img, 0, 0, x*3, y*3))

