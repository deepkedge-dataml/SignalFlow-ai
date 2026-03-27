import cv2


def generate_overlay(image_path, heatmap_path, alpha=0.4):
    image = cv2.imread(image_path)
    heatmap = cv2.imread(heatmap_path, cv2.IMREAD_GRAYSCALE)

    print("Image shape:", image.shape if image is not None else None)
    print("Heatmap shape:", heatmap.shape if heatmap is not None else None)

    heatmap_resized = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
    colored_heatmap = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)

    overlay = cv2.addWeighted(image, 1 - alpha, colored_heatmap, alpha, 0)
    print("Resized heatmap shape:", heatmap_resized.shape)

    return overlay


