import matplotlib.pyplot as plt
import numpy as np
import imageio
import cv2


def raw_distribution(fname, all_alphas, all_ks, all_seg_lengths):
    plt.figure(f'raw_distribution')
    plt.scatter(all_alphas, all_ks,
                c=np.minimum(np.ones_like(all_seg_lengths), all_seg_lengths / np.quantile(all_seg_lengths, 0.75)),
                s=0.3, alpha=0.7)
    plt.colorbar()
    plt.savefig(f'{fname}')
    plt.close('all')


def sample_distribution(image_path, stack_samples, all_seg_lengths, cluster, cluster_states):
    sample_distrib_file = f'{image_path}/data_filtered_distribution.png'
    plt.figure(f'data cluster_center and sample distribution')
    plt.scatter(stack_samples[:, 0], stack_samples[:, 1],
                c=np.minimum(np.ones_like(all_seg_lengths), all_seg_lengths / np.quantile(all_seg_lengths, 0.75)),
                s=0.3, alpha=0.7)
    for i, cluster_mean in enumerate(cluster.means_):
        if cluster_states[i] == 0:
            c_color = 'red'
        elif cluster_states[i] == 1:
            c_color = 'orange'
        elif cluster_states[i] == 2:
            c_color = 'green'
        else:
            c_color = 'blue'
        plt.scatter(cluster_mean[0], cluster_mean[1], c=c_color, marker='+', zorder=10)
    plt.colorbar()
    plt.savefig(sample_distrib_file)
    plt.close('all')


def cluster_boundary(image_path, all_alphas, all_ks, cluster):
    boundary_file = f'{image_path}/data_mixture_boundary.png'
    alpha_range = np.linspace(-2.2, 4.2, 100)
    logk_range = np.linspace(-4.0, 4.0, 150)
    H, xedges, yedges = np.histogram2d(all_ks, all_alphas, bins=[logk_range, alpha_range])
    plt.figure(figsize=(20, 16), dpi=256)
    plt.title("Mixture boundary", fontsize=28)
    xx, yy = np.meshgrid(alpha_range, logk_range)
    Z = np.stack((xx, yy)).T.reshape(-1, 2)
    ZC = cluster.predict(Z)
    plt.scatter(Z[:, 0], Z[:, 1], c=ZC, s=10, alpha=0.5, zorder=1)
    plt.imshow(H, extent=(alpha_range[0], alpha_range[-1], logk_range[0], logk_range[-1]), origin='lower', alpha=0.9)
    for i, cluster_mean in enumerate(cluster.means_):
        label = cluster.predict([[cluster_mean[0], cluster_mean[1]]])
        plt.scatter(cluster_mean[0], cluster_mean[1], c='w', marker='${}$'.format(label), zorder=10, s=1000)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.xlabel(f'alpha', fontsize=24)
    plt.ylabel(f'logK', fontsize=24)
    plt.savefig(boundary_file)
    plt.close('all')


def cps_visualization(image_path, xs, ys, cps, alpha=0.8, ext=50):
    assert len(xs) == len(ys)
    rgba_images = []
    x_width = int(np.max(xs) - np.min(xs) + ext)
    y_width = int(np.max(ys) - np.min(ys) + ext)
    image_size = (max(x_width, y_width), max(x_width, y_width), 3)
    image = np.zeros(image_size, dtype=np.uint8)
    rgba_image = cv2.cvtColor(image, cv2.COLOR_RGB2RGBA)
    rgba_image[:, :, 3] = 0
    cps = cps[1:-1]
    cps_set = set(np.array([[cp-1, cp, cp+1] for cp in cps]).flatten())
    cps_rad = {}
    for cp in cps:
        for i, cpk in enumerate(range(cp-1, cp+2)):
            cps_rad[cpk] = int(i*1 + 2)

    for i in range(len(xs) - 1):
        line_overlay = cv2.line(rgba_image, (int(xs[i] - np.min(xs) + ext//2), int(ys[i] - np.min(ys) + ext//2)),
                              (int(xs[i+1] - np.min(xs) + ext//2), int(ys[i+1] - np.min(ys) + ext//2)),
                              (255, 255, 0, 255), 1)
        rgba_image = cv2.addWeighted(line_overlay, alpha, rgba_image, 0, 0)
        if i in cps_set:
            circle_overlay = cv2.circle(rgba_image, center=(int(xs[i] - np.min(xs) + ext//2), int(ys[i] - np.min(ys) + ext//2)),
                                        radius=cps_rad[i], color=(255, 0, 0, 255))
            image_new = cv2.addWeighted(circle_overlay, alpha, rgba_image, 0, 0)
            rgba_images.append(image_new)
        else:
            rgba_images.append(rgba_image.copy())

    with imageio.get_writer(f'{image_path}.gif', mode='I') as writer:
        for i in range(len(rgba_images)):
            writer.append_data(np.array(rgba_images[i]))
