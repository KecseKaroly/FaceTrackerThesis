import torch
import torch.backends.cudnn as cudnn
import numpy as np
from tools.retinaface import RetinaFace
from utils.box_utils import decode, decode_landm
from torchvision.ops import nms
from layers.functions.prior_box import PriorBox


class FaceDetector:
    def __init__(self, cfg, model_path, confidence_threshold=0.55, nms_threshold=0.4):
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if(self.device == 'cuda'):
            print("CUDA AVAILABLE")
        self.conf_threshold = confidence_threshold
        self.nms_threshold = nms_threshold

        net = RetinaFace(cfg=cfg, phase='test')
        net = self.load_model(net, model_path, load_to_cpu=False)
        net.eval()
        cudnn.benchmark = True

        self.priorbox = PriorBox(cfg, image_size=(720, 1280))
        self.priors = self.priorbox.forward().to(self.device)
        self.model = net.to(self.device)

    def detect_faces(self, frame, timer): 
        resize = 1
        device = self.device
        cfg = self.cfg

        img = frame.astype(np.float32)
        scale = torch.Tensor([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
        img -= (104, 117, 123)
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img).unsqueeze(0)
        img = img.to(device)
        scale = scale.to(device)

        timer.tic()
        loc, conf, _ = self.model(img)
        timer.tic()

        
        boxes = decode(loc.data.squeeze(0), self.priors, cfg['variance'])
        boxes = boxes * scale / resize
        boxes = boxes.cpu().numpy()
        scores = conf.squeeze(0).data.cpu().numpy()[:, 1]
        #landms = decode_landm(landms.data.squeeze(0), priors, cfg['variance'])
        scale1 = torch.Tensor([img.shape[3], img.shape[2]] * 5).to(device)
        #landms = landms * scale1 / resize
        #landms = landms.cpu().numpy()

        inds = np.where(scores > self.conf_threshold)[0]
        boxes = boxes[inds]
        #landms = landms[inds]
        scores = scores[inds]
        
        order = scores.argsort()[::-1]
        boxes = boxes[order]
        #landms = landms[order]
        scores = scores[order]

        dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
        
        keep = nms(torch.tensor(boxes), torch.tensor(scores), self.nms_threshold).numpy()
        dets = dets[keep]
        #landms = landms[keep]


        img_info = {"id": 0}
        height, width = frame.shape[:2]
        img_info["height"] = height
        img_info["width"] = width
        img_info["raw_img"] = frame.copy()
        return dets, img_info

    @staticmethod
    def check_keys(model, pretrained_state_dict):
        ckpt_keys = set(pretrained_state_dict.keys())
        model_keys = set(model.state_dict().keys())
        used_pretrained_keys = model_keys & ckpt_keys
        assert len(used_pretrained_keys) > 0, 'load NONE from pretrained checkpoint'
        return True

    @staticmethod
    def remove_prefix(state_dict, prefix):
        ''' Old style model is stored with all names of parameters sharing common prefix 'module.' '''
        f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
        return {f(key): value for key, value in state_dict.items()}


    def load_model(self, model, pretrained_path, load_to_cpu):
        print('Loading pretrained model from {}'.format(pretrained_path))
        if load_to_cpu:
            pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage)
        else:
            device = torch.cuda.current_device()
            pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage.cuda(device))
        if "state_dict" in pretrained_dict.keys():
            pretrained_dict = self.remove_prefix(pretrained_dict['state_dict'], 'module.')
        else:
            pretrained_dict = self.remove_prefix(pretrained_dict, 'module.')
        self.check_keys(model, pretrained_dict)
        model.load_state_dict(pretrained_dict, strict=False)
        return model
