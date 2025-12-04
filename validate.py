from pathlib import Path
from anomalib.engine import Engine
from anomalib.models import Patchcore
from anomalib.data import Folder
import torch


def find_latest_checkpoint():
    """Find the most recent checkpoint file."""
    results_dir = Path("results")
    if not results_dir.exists():
        raise FileNotFoundError("No 'results' directory found. Train the model first.")
    
    checkpoints = list(results_dir.glob("**/weights/lightning/model.ckpt"))
    if not checkpoints:
        checkpoints = list(results_dir.glob("**/*.ckpt"))
    
    if not checkpoints:
        raise FileNotFoundError("No checkpoint files found in results directory.")
    
    # Return the most recently modified checkpoint
    latest = max(checkpoints, key=lambda p: p.stat().st_mtime)
    print(f"Using checkpoint: {latest}")
    return latest


def main():
    # Setup datamodule (same as training)
    datamodule = Folder(
        name="wzor_pasek_A",
        root="./dataset/wzor_pasek_A",
        normal_dir="train/good",
        abnormal_dir="test/defect",
        normal_test_dir="test/good",
        normal_split_ratio=0.2,
        extensions=[".PNG"],
        train_batch_size=32,
        eval_batch_size=32,
        num_workers=8
    )

    # Find and load the trained model
    checkpoint_path = find_latest_checkpoint()
    
    # Create model and engine
    model = Patchcore.load_from_checkpoint(checkpoint_path)
    engine = Engine()
    
    # Run test evaluation
    print("\n" + "="*50)
    print("Running evaluation on test set...")
    print("="*50 + "\n")
    
    test_results = engine.test(model=model, datamodule=datamodule)
    
    # Print results
    print("\n" + "="*50)
    print("TEST RESULTS")
    print("="*50)
    for result in test_results:
        for key, value in result.items():
            print(f"{key}: {value:.4f}" if isinstance(value, float) else f"{key}: {value}")
    
    # Run prediction on ALL images in test folders
    print("\n" + "="*50)
    print("Running predictions on ALL test images...")
    print("="*50 + "\n")
    
    # Get ALL test images from both folders
    test_defect_dir = Path("./dataset/wzor_pasek_A/test/defect")
    test_good_dir = Path("./dataset/wzor_pasek_A/test/good")
    
    defect_images = sorted(test_defect_dir.glob("*.PNG"))
    good_images = sorted(test_good_dir.glob("*.PNG"))
    
    print(f"Found {len(defect_images)} defect images and {len(good_images)} good images")
    print(f"Total: {len(list(defect_images)) + len(list(good_images))} images\n")
    
    # Collect all results
    results = []
    
    # Helper function to run predictions on a folder
    def predict_folder(folder_path, ground_truth_label):
        # Use engine.predict with data_path parameter
        predictions = engine.predict(
            model=model, 
            data_path=str(folder_path),
            ckpt_path=str(checkpoint_path)
        )
        
        folder_results = []
        if predictions:
            for batch in predictions:
                image_paths = getattr(batch, "image_path", None)
                pred_scores = getattr(batch, "pred_score", None)
                pred_labels = getattr(batch, "pred_label", None)
                
                if image_paths is None:
                    continue
                    
                num_samples = len(image_paths) if image_paths else 0
                for i in range(num_samples):
                    img_path = image_paths[i] if image_paths else f"image_{i}"
                    
                    if pred_scores is not None:
                        score = pred_scores[i].item() if torch.is_tensor(pred_scores[i]) else float(pred_scores[i])
                    else:
                        score = None
                    
                    if pred_labels is not None:
                        label = "ANOMALY" if pred_labels[i] else "NORMAL"
                    else:
                        label = "N/A"
                    
                    if score is not None:
                        folder_results.append((Path(img_path).name, score, label, ground_truth_label))
        return folder_results
    
    # Predict on defect folder
    print("Predicting on defect images...")
    results.extend(predict_folder(test_defect_dir, "DEFECT"))
    
    # Predict on good folder  
    print("Predicting on good images...")
    results.extend(predict_folder(test_good_dir, "GOOD"))
    
    # Print results grouped by folder
    print("DEFECT IMAGES (should be ANOMALY):")
    print("-" * 60)
    defect_results = [r for r in results if r[3] == "DEFECT"]
    for name, score, pred, gt in sorted(defect_results):
        status = "✓" if pred == "ANOMALY" else "✗"
        print(f"  {status} {name}: Score={score:.4f}, Prediction={pred}")
    
    correct_defects = sum(1 for r in defect_results if r[2] == "ANOMALY")
    print(f"\nDefect Detection: {correct_defects}/{len(defect_results)} correct")
    
    print("\nGOOD IMAGES (should be NORMAL):")
    print("-" * 60)
    good_results = [r for r in results if r[3] == "GOOD"]
    for name, score, pred, gt in sorted(good_results):
        status = "✓" if pred == "NORMAL" else "✗"
        print(f"  {status} {name}: Score={score:.4f}, Prediction={pred}")
    
    correct_goods = sum(1 for r in good_results if r[2] == "NORMAL")
    print(f"\nGood Detection: {correct_goods}/{len(good_results)} correct")
    
    # Overall accuracy
    total_correct = correct_defects + correct_goods
    total = len(results)
    print(f"\n{'='*60}")
    print(f"OVERALL ACCURACY: {total_correct}/{total} ({100*total_correct/total:.1f}%)")
    
    print("\n" + "="*50)
    print("Validation complete!")
    print("="*50)


if __name__ == "__main__":
    main()

