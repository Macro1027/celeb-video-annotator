import os
import torch
import cv2
from PIL import Image
from tqdm import tqdm


class ExtractFeaturesMTCNN:
    def __init__(self, mtcnn, resnet, device='cpu', output_size=224):
        self.mtcnn = mtcnn
        self.output_size = output_size
        self.path = 'data/Original Images/Original Images/'  # Updated path
        self.resnet = resnet
        self.device = device

    def detect_and_crop(self, img, draw=False):
        with torch.no_grad():
            boxes, probs, landmarks = self.mtcnn.detect(img, landmarks=True)
            print("Face detection completed")
            # Check if any faces were detected
            if boxes is None or len(boxes) == 0:
                return None, None

            # Extract first face's detection results
            results = {
                'boxes': boxes[0],
                'probs': probs[0],
                'landmarks': landmarks[0]
            }

            # Get face crop - also check if cropping succeeds
            faces = self.mtcnn(img)
            if faces is None or len(faces) == 0:
                return results, None

            face = faces[0]
        return results, face

    def process_image_from_path(self, path):
        img = cv2.imread(path)
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        return img

    def align_and_embed_from_df(self, df, label, out='results/aligned-faces/'):
        print(f"Extracting images of {label}")
        db_entry = []
        out += f"{label}"
        df_individual = df.loc[df['label'] == label]
        for i in tqdm(range(1, len(df_individual))):
            read_path = f"{self.path}{label}/{df_individual.iloc[i]['id']}"
            write_path = f"{out}/{label}_{i}.jpg"

            img_processed = self.process_image_from_path(read_path)
            results, aligned = self.detect_and_crop(img_processed)
            if aligned is None or results['boxes'] is None:
                print(f"Skipping image {i}, no face found")
                continue

            aligned = aligned[[2,1,0], :, :]

            os.makedirs(os.path.dirname(write_path), exist_ok=True)

            cv2.imwrite(write_path, aligned.permute(1,2,0).numpy())
            # expand from (C, H, W) to (1, C, H, W)
            aligned_tensor = aligned.unsqueeze(0).to(self.device)
            face_id = f"{label}_photo_{i}"
            embedding = self.resnet(aligned_tensor.half()).squeeze(0)
            face_metadata = {'person_id': label, 'image_id': str(i)}
            db_entry.append((face_id, embedding, face_metadata))

            del aligned, img_processed, results, face_id, embedding, face_metadata
            torch.cuda.empty_cache()
        return db_entry 