if __name__ == '__main__':
    from tran import *
else:
    from .tran import *

def pil_rotate(pic, angle):
    _pic=dict()
    for key in pic.keys():
        _pic[key] = pic[key].rotate(angle)
    return _pic

class PILRandomRotation(object):
    def __call__(self, pic):
        angle = random.randint(0,12)*15
        return imgs2arrs(pil_rotate(arrs2imgs(pic), angle))

def pil_tran(pic, tran=None):
    if tran is None:
        return pic
    if isinstance(tran, list):
        for t in tran:
            for key in pic.keys():
                pic[key] = pic[key].transpose(t)
    else:
        for key in pic.keys():
            pic[key] = pic[key].transpose(tran)
    return pic

class Aug4Val(object):
    number = 8
    @staticmethod
    def forward_val(pic, flag):
        flag %= Aug4Val.number
        if flag==0:
            return pic
        pic = arrs2imgs(pic)
        if flag==1:
            return imgs2arrs(pil_tran(pic, tran=Image.FLIP_LEFT_RIGHT))
        if flag==2:
            return imgs2arrs(pil_tran(pic, tran=Image.FLIP_TOP_BOTTOM))
        if flag==3:
            return imgs2arrs(pil_tran(pic, tran=Image.ROTATE_180))
        if flag==4:
            return imgs2arrs(pil_tran(pic, tran=[Image.FLIP_LEFT_RIGHT,Image.FLIP_TOP_BOTTOM]))
        if flag==5:
            return imgs2arrs(pil_tran(pic, tran=[Image.ROTATE_180,Image.FLIP_TOP_BOTTOM]))
        if flag==6:
            return imgs2arrs(pil_tran(pic, tran=[Image.ROTATE_180,Image.FLIP_LEFT_RIGHT]))
        if flag==7:
            return imgs2arrs(pil_tran(pic, tran=[Image.ROTATE_180,Image.FLIP_LEFT_RIGHT,Image.FLIP_TOP_BOTTOM]))


