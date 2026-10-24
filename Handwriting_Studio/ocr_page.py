# Heuristic line separation + trained TrOCR. Clean single-column pages only.
import argparse,json
from pathlib import Path
import cv2,numpy as np,torch
from PIL import Image,ImageOps
from transformers import TrOCRProcessor,VisionEncoderDecoderModel
def prep(im):
    im=ImageOps.exif_transpose(im).convert('RGBA'); bg=Image.new('RGBA',im.size,'white'); bg.alpha_composite(im)
    return ImageOps.autocontrast(bg.convert('L'),cutoff=.2).convert('RGB')
def segment(im):
    im=prep(im); im.thumbnail((2400,3400)); gray=np.array(im.convert('L'))
    ink=cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)[1]
    horizontal=cv2.morphologyEx(ink,cv2.MORPH_OPEN,np.ones((1,max(80,im.width//3)),np.uint8))
    vertical=cv2.morphologyEx(ink,cv2.MORPH_OPEN,np.ones((max(80,im.height//3),1),np.uint8))
    mask=cv2.subtract(cv2.subtract(ink,horizontal),vertical)
    def rotate(a,angle,fill=0):
        h,w=a.shape[:2]; return cv2.warpAffine(a,cv2.getRotationMatrix2D((w/2,h/2),angle,1),(w,h),borderValue=fill)
    small=cv2.resize(mask,(max(1,im.width//3),max(1,im.height//3)))
    angles=np.arange(-5,5.01,.5); scores=[np.square(rotate(small,a).sum(axis=1,dtype=np.float64)).sum() for a in angles]
    angle=float(angles[int(np.argmax(scores))]) if mask.any() else 0.
    mask=rotate(mask,angle); im=Image.fromarray(rotate(np.array(im),angle,(255,255,255)))
    n,labels,stats,_=cv2.connectedComponentsWithStats(mask); clean=np.zeros_like(mask)
    for i in range(1,n):
        if stats[i,cv2.CC_STAT_AREA]>=6: clean[labels==i]=255
    active=(np.count_nonzero(clean,axis=1)>=max(2,int(im.width*.001))).astype(np.uint8)
    active=cv2.morphologyEx(active[:,None],cv2.MORPH_CLOSE,np.ones((7,1),np.uint8)).ravel()
    edges=np.diff(np.pad(active.astype(int),(1,1))); boxes=[]
    for y0,y1 in zip(np.flatnonzero(edges==1),np.flatnonzero(edges==-1)):
        xs=np.flatnonzero(clean[y0:y1].any(axis=0))
        if y1-y0>=5 and len(xs)>=8: boxes.append([max(0,int(xs[0])-8),max(0,int(y0)-5),min(im.width,int(xs[-1])+9),min(im.height,int(y1)+5)])
    return im,boxes,angle
@torch.inference_mode()
def transcribe(image,model_path,out):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    device='cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
    p=TrOCRProcessor.from_pretrained(model_path,use_fast=False); m=VisionEncoderDecoderModel.from_pretrained(model_path).to(device).eval()
    im,boxes,angle=segment(Image.open(image));im.save(out/'deskewed.png');lines=[]
    for i,box in enumerate(boxes):
        crop=im.crop(box);crop.save(out/f'line_{i:03}.png')
        px=p(images=prep(crop),return_tensors='pt').pixel_values.to(device)
        text=p.batch_decode(m.generate(px,max_new_tokens=128,num_beams=1),skip_special_tokens=True)[0]
        lines.append({'line':i,'box_on_deskewed_image':box,'text':text})
    text='\n'.join(r['text'] for r in lines);(out/'transcription.txt').write_text(text,encoding='utf-8')
    (out/'layout.json').write_text(json.dumps({'rotation_degrees':angle,'lines':lines},indent=2))
    return text
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('image');p.add_argument('--model',default='model');p.add_argument('--out',default='page_output');a=p.parse_args()
    print(transcribe(a.image,a.model,a.out))
