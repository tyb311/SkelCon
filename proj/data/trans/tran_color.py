if __name__ == '__main__':
    from tran import *
else:
    from .tran import *


def random_channel(rgb, tran=None):##cv2.COLOR_RGB2HSV,HSV不好#
    if tran is None:
        tran = random.choice([
            cv2.COLOR_RGB2GRAY, cv2.COLOR_RGB2BGR, 
            cv2.COLOR_RGB2XYZ, cv2.COLOR_RGB2LAB,
            cv2.COLOR_RGB2HLS, cv2.COLOR_RGB2LUV,        
            cv2.COLOR_RGB2YCrCb, cv2.COLOR_RGB2YUV,
            ])
    # if rgb.shape[-1]!=3:#单通道图片不做变换
    #     return rgb
    rgb = cv2.cvtColor(rgb, tran)
    if tran==cv2.COLOR_RGB2LAB:
        rgb = cv2.split(rgb)[0]
    elif tran==cv2.COLOR_RGB2XYZ:
        rgb = cv2.split(rgb)[0]
    elif tran==cv2.COLOR_RGB2LUV:
        rgb = cv2.split(rgb)[0]
    elif tran==cv2.COLOR_RGB2HLS:
        rgb = cv2.split(rgb)[1]
    elif tran==cv2.COLOR_RGB2YCrCb:
        rgb = cv2.split(rgb)[0]
    elif tran==cv2.COLOR_RGB2YUV:
        rgb = cv2.split(rgb)[0]
    elif tran==cv2.COLOR_RGB2BGR:
        rgb = cv2.split(rgb)[1]
        # rgb = random.choice(cv2.split(rgb))#
    return rgb

class Aug4CSA(object):#Color Space Augment
    number = 8
    trans = [
            cv2.COLOR_RGB2GRAY, cv2.COLOR_RGB2BGR, 
            cv2.COLOR_RGB2XYZ, cv2.COLOR_RGB2LAB,
            cv2.COLOR_RGB2HLS, cv2.COLOR_RGB2LUV,        
            cv2.COLOR_RGB2YCrCb, cv2.COLOR_RGB2YUV,
            ]
    # trans_test = [
    #         cv2.COLOR_RGB2GRAY, cv2.COLOR_RGB2BGR, 
    #         cv2.COLOR_RGB2XYZ, cv2.COLOR_RGB2LAB,
    #         ]
    @staticmethod
    def forward_val(pic, flag):
        flag %= Aug4CSA.number
        pic['img'] = random_channel(pic['img'], tran=Aug4CSA.trans[flag])
        return pic
    @staticmethod
    def forward_train(pic):  #random channel mixture
        a = random_channel(pic['img'])
        b = random_channel(pic['img'])
        alpha = random.random()#/2+0.1
        pic['img'] = (alpha*a + (1-alpha)*b).astype(np.uint8)
        return pic
    @staticmethod
    def forward_test(pic):
        pic['csa'] = np.concatenate([random_channel(pic['img'], t) for t in Aug4CSA.trans])
        return pic
#end#

