
import numpy as np

def srgb2linear(s):
    if s <= 0.04045:
        return s / 12.92
    return float(np.power((s+0.055)/1.055, 2.4))

def linear2srgb(l):
    if l <= 0.0031308:
        return 12.92*l
    return 1.055*np.power(l, 1/2.4) - 0.055

def s2l(s):
    return list(map(srgb2linear, s))

def l2s(l):
    return list(map(linear2srgb, l))

def lumen(l):
    s = [a*b for a,b in zip(l, [0.2126, 0.7152, 0.0722])]
    return sum(s)

def srgb2hex(s):
    return hex(int(round(s*255.0)))[2:]

def hex2srgb(h):
    return int(h, 16)/255.0

def h2s(h):
    return [hex2srgb(h[2*n:2*n+2]) for n in range(3)]

def s2h(s):
    out = "#"
    for c in map(srgb2hex, s):
        out += c.zfill(2)
    return out

def linearinterp(a, b, x):
    return (b-a)*x + a

def lerp(a, b, x):
    return list(map(linearinterp, a,b,x))

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    arr = []

    bg = h2s('212121')
    txt = lerp(bg, [1]*3, [0.05]*3)
    lbg = s2l(bg)
    ltxt = s2l(txt)

    mix = [lumen(ltxt), lumen(lbg)]
    for m in [mix[0]] + [(mix[0] + mix[1]) / 2.0] + [mix[-1]]:
        bright = m
        #bright = lumen(ltxt)
        #bright = (lumen(lbg) + lumen(ltxt)) / 2.0
        #bright = pow(10, (np.log10(lumen(lbg)) + np.log10(lumen(ltxt))) / 2.0)
        rd = s2l(h2s("FF0000"))
        scale = bright / lumen(rd)
        rd2 = [c*scale for c in rd]

        arr += [l2s(rd2)]
        
        print(s2h(txt))
        print(s2h(l2s(rd2)))
        print('#212121')

    plt.imshow(np.array([[txt, n, h2s('212121')] for n in arr], dtype=np.float32))
    plt.show()

       



    
