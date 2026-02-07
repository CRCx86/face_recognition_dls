import torch
from face_recognition.heatmap_dataset import CelebAHeatmapDataset

""""""
" ТЕСТЫ ДЛЯ СЕБЯ "
""""""


dataset = CelebAHeatmapDataset(
    "E:\\Deep Learning School\\FR\\result\\cropped"
)
loader = torch.utils.data.DataLoader(
    dataset, batch_size=64, shuffle=True, num_workers=4
)


def heatmaps_to_points(hm):
    # hm: [N, H, W]
    pts = []
    for i in range(hm.shape[0]):
        idx = torch.argmax(hm[i])
        y = idx // hm.shape[2]
        x = idx % hm.shape[2]
        pts.append((x.item(), y.item()))

    return pts


def mean_pixel_error(pred_hm, gt_hm):
    err = 0
    for i in range(pred_hm.shape[0]):
        p = torch.argmax(pred_hm[i])
        g = torch.argmax(gt_hm[i])
        px, py = p % pred_hm.shape[2], p // pred_hm.shape[2]
        gx, gy = g % gt_hm.shape[2], g // gt_hm.shape[2]
        err += torch.sqrt((px - gx)**2 + (py - gy)**2)
    return err / pred_hm.shape[0]


def run_test(model):
    model.load_state_dict(torch.load("models/hourglass/hourglass_landmarks_best.pth"))
    model.eval()

    img, gt_hm = dataset[0]
    img_path = dataset.images[0]

    with torch.no_grad():
        pred_hm = model(img.unsqueeze(0).cuda())[-1][0].cpu()

    pred_pts = heatmaps_to_points(pred_hm)

    print("Файл:", img_path.name, "Predicted points:", pred_pts)


def run_tests(model):
    model.load_state_dict(torch.load("models/hourglass/hourglass_landmarks_best.pth"))

    model.eval()
    errors = []

    with torch.no_grad():
        for imgs, gt_hms in loader:
            imgs = imgs.cuda()
            gt_hms = gt_hms.cuda()

            pred_hms = model(imgs)[-1]

            for i in range(imgs.size(0)):
                e = mean_pixel_error(pred_hms[i], gt_hms[i])
                errors.append(e.item())

    print("Mean Pixel Error:", sum(errors) / len(errors))
