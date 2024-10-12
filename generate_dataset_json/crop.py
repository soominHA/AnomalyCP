import os
import json

# 데이터셋 주소 : https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=realm&dataSetSn=153

class CropSolver(object):
    CLSNAMES = ['cucumber', 'eggplant', 'pumkin', 'sweet_pumkin']

    def __init__(self, root='data/crop'):
        self.root = root
        self.meta_path = f'{root}/meta.json'

    def run(self):
        info = dict(train={}, test={})
        anomaly_samples = 0
        normal_samples = 0
        for cls_name in self.CLSNAMES:
            cls_dir = f'{self.root}/{cls_name}'
            for phase in ['train', 'test']:
                cls_info = []
                species = os.listdir(f'{cls_dir}/{phase}')
                for specie in species:
                    is_abnormal = True if specie not in ['good'] else False
                    img_names = os.listdir(f'{cls_dir}/{phase}/{specie}')
                    img_names.sort()
                    for idx, img_name in enumerate(img_names):
                        info_img = dict(
                            img_path=f'{cls_dir}/{phase}/{specie}/{img_name}',
                            cls_name=cls_name,
                            mask_path="",
                            specie_name=specie,
                            anomaly=1 if is_abnormal else 0,
                        )
                        cls_info.append(info_img)
                        if is_abnormal:
                            anomaly_samples = anomaly_samples + 1
                        else:
                            normal_samples = normal_samples + 1
                info[phase][cls_name] = cls_info
        with open(self.meta_path, 'w') as f:
            f.write(json.dumps(info, indent=4) + "\n")
        print('normal_samples', normal_samples, 'anomaly_samples', anomaly_samples)

if __name__ == '__main__':
    runner = CropSolver(root='C:/Users/dmkwo/Documents/soo/AnomalyCLIP/data/crop')
    runner.run()
