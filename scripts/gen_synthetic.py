import argparse, os
from pathlib import Path
import numpy as np, cv2, random

def gen_img(w=256,h=256):
    img = np.zeros((h,w,3), np.uint8)
    # random rectangles/circles to mimic tools/anatomy shapes
    for _ in range(random.randint(1,5)):
        if random.random()<0.5:
            x1,y1 = random.randint(0,w-50), random.randint(0,h-50)
            x2,y2 = x1+random.randint(10,80), y1+random.randint(10,80)
            cv2.rectangle(img,(x1,y1),(x2,y2),(random.randint(50,255),)*3, -1)
        else:
            cx,cy = random.randint(30,w-30), random.randint(30,h-30)
            r = random.randint(10,40)
            cv2.circle(img,(cx,cy),r,(random.randint(50,255),)*3, -1)
    # add lighting gradients/noise
    img = cv2.GaussianBlur(img,(0,0), random.uniform(0.5,1.5))
    return img

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', type=str, required=True)
    ap.add_argument('--n', type=int, default=200)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for i in range(args.n):
        img = gen_img()
        cv2.imwrite(str(out / f'frame_{i:04d}.png'), img)

if __name__ == '__main__':
    main()
