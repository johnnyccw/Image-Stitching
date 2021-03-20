import cv2
import numpy as np
from sklearn.neighbors import KDTree

def cylidical_project(img, f):
    img = img.astype(np.float32)
    h_orig = img.shape[0]
    w_orig = img.shape[1]
    h = h_orig
    w = np.ceil(2 * f * np.arctan2(w_orig / 2, f)).astype(np.int)
    res = np.zeros((h, w, 3), dtype=np.float32)
    for i in range(h):
        for j in range(w):
            x = j - w // 2
            y = i - h // 2
            x_orig = f * np.tan(x / f)
            y_orig = (x_orig * x_orig + f * f) ** 0.5 * y / f
            j_orig = (x_orig + w_orig / 2).astype(np.int)
            i_orig = (y_orig + h_orig / 2).astype(np.int)
            if i_orig >= 0 and j_orig >= 0 and i_orig < h_orig and j_orig < w_orig:
                res[i, j, :] = img[i_orig, j_orig, :]
    return res.astype(np.uint8)

def find_feature(img, scale):
    k = 5
    s = 3
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img = cv2.resize(img, (img.shape[1] // scale, img.shape[0] // scale))
    img = img.astype(np.float32)
    img = cv2.GaussianBlur(img, (k, k), s)
    Iy, Ix = np.gradient(img)
    Ix2 = Ix ** 2
    Iy2 = Iy ** 2
    Ixy = Ix * Iy
    Sx2 = cv2.GaussianBlur(Ix2, (k, k), s)
    Sy2 = cv2.GaussianBlur(Iy2, (k, k), s)
    Sxy = cv2.GaussianBlur(Ixy, (k, k), s)
    detM = (Sx2 * Sy2) - (Sxy ** 2)
    traceM = Sx2 + Sy2
    R = detM - 0.05 * (traceM ** 2)
    R[0 : 160 // scale, :], R[-160 // scale : R.shape[0], :], R[:, 0 : 20 // scale], R[:, -20 // scale : R.shape[1]] = 0, 0, 0, 0
    _,R = cv2.threshold(R, R.mean() + R.std(), R.max(), cv2.THRESH_TOZERO)
    for patch_size in [20]:
        for i in range(R.shape[0] // patch_size - 1):
            for j in range(R.shape[1] // patch_size - 1):
                patch = R[i * patch_size : (i+1) * patch_size, j * patch_size : (j + 1) * patch_size]
                max_idx = patch.argmax()
                for k in range(patch_size):
                    for l in range(patch_size):
                        if k != max_idx // patch_size or l != max_idx % patch_size:
                            R[i * patch_size + k, j * patch_size + l] = 0
    i, j = np.where(R > 0)
    feature_loc = np.array([i * scale, j * scale], dtype=np.int)
    feature_loc = np.swapaxes(feature_loc, 0 , 1)
    return feature_loc

def descript_feature(img, feature_loc):
    img = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    patch_size = 8
    description = []
    for loc in feature_loc:
        patch = img[loc[0] - patch_size // 2 : loc[0] + patch_size // 2, loc[1] - patch_size // 2 : loc[1] + patch_size // 2]
        description.append(patch.flatten())
    description = np.array(description, dtype=np.float32)
    return description

def ransac(loc1, loc2, des1, des2):
    kdt = KDTree(des2)
    _, match_idx = kdt.query(des1)
    match_idx = np.squeeze(match_idx)
    shift = np.zeros_like(loc1)
    for i in range(loc1.shape[0]):
        shift[i] = loc1[i] - loc2[match_idx[i]]
    best_score = 0
    best_shift = shift[0]
    for s in shift:
        score = 0
        for k in shift:
            if abs(k[0] - s[0]) < 3 and abs(k[1] - s[1]) < 3:
                score += 1
        if score > best_score:
            best_score = score
            best_shift = s
    return best_shift
            
def stitch(img1, img2, shift):
    img1 = img1.astype(np.int)
    img2 = img2.astype(np.int)
    h1 = img1.shape[0]
    h2 = img2.shape[0]
    w1 = img1.shape[1]
    w2 = img2.shape[1]
    img1 = np.hstack((img1, np.zeros((h1, shift[1], 3))))
    img2 = np.hstack((np.zeros((h2, w1 - w2 + shift[1], 3)), img2))
    if shift[0] >= 0:
        img1 = np.vstack((img1, np.zeros((shift[0], img1.shape[1], 3))))
        img2 = np.vstack((np.zeros((h1 - h2 + shift[0], img2.shape[1], 3)), img2))
    else:
        img2 = np.vstack((np.zeros((h1 - h2 + shift[0], img2.shape[1], 3)), img2, np.zeros((-shift[0], img2.shape[1], 3))))
    res = np.zeros_like(img1, dtype=np.int)
    for j in range(res.shape[1]):
        if j < w1 - w2 + shift[1]:
            res[:, j] = img1[:, j]
        elif j < w1:
            for i in range(res.shape[0]):
                if img1[i, j].all()==0:
                    img1[i, j] = img2[i, j]
                if img2[i, j].all()==0:
                    img2[i, j] = img1[i, j]
            res[:, j] = (w1 - j) / (w2 - shift[1]) * img1[:, j] + (j - w1 + w2 - shift[1]) / (w2 - shift[1]) * img2[:, j]
        else:
            res[:, j] = img2[:, j]
    res = res.astype(np.uint8)
    return res

def rectangling(img):
    tl, tr, bl, br = 0,0,0,0
    for i in range(img.shape[0]):
        if img[i, 0].any()!=0:
            tl = i
            break
    for i in range(img.shape[0]):
        if img[i, img.shape[1] - 1].any() != 0:
            tr = i
            break
    for i in range(img.shape[0] - 1, -1, -1):
        if img[i, 0].any()!=0:
            bl = i
            break
    for i in range(img.shape[0] - 1, -1, -1):
        if img[i, img.shape[1] - 1].any() != 0:
            br = i
            break
    pts1 = np.float32([[0, tl], [0, bl], [img.shape[1], br], [img.shape[1], tr]])
    pts2 = np.float32([[0,0], [0, bl - tl], [img.shape[1], bl - tl], [img.shape[1], 0]])
    M=cv2.getPerspectiveTransform(pts1, pts2)
    dst=cv2.warpPerspective(img,M,(img.shape[1], bl - tl))
    return dst.astype(np.uint8)

def main():
    N = 6
    cylids = []
    feature_locs = []
    descriptions = []
    for i in range(N):
        img = cv2.imread('img/' + str(i + 1) + '.jpg')
        cylid = cylidical_project(img, 1672)
        feature_loc = find_feature(cylid, 1)
        description = descript_feature(cylid, feature_loc)
        cylids.append(cylid)
        feature_locs.append(feature_loc)
        descriptions.append(description)
    pano = cylids[0]
    for i in range(N - 1):
        shift = ransac(feature_locs[i], feature_locs[i + 1], descriptions[i], descriptions[i + 1])
        pano = stitch(pano, cylids[i + 1], shift)
    cv2.imwrite('pano_orig.jpg', pano)
    pano = rectangling(pano)
    cv2.imwrite('pano_rect.jpg', pano)

if __name__== "__main__":
    main()
