"""
Test script to verify all improvements are working correctly
Run this before doing full experiments
"""
import numpy as np
import sys

def test_imports():
    """Test that all required packages are available"""
    print("\n" + "="*60)
    print("TEST 1: Checking Imports")
    print("="*60)
    
    try:
        import sklearn
        print("✓ scikit-learn:", sklearn.__version__)
    except ImportError:
        print("✗ scikit-learn not found")
        return False
    
    try:
        import clip
        print("✓ CLIP installed")
    except ImportError:
        print("✗ CLIP not found. Install with: pip install git+https://github.com/openai/CLIP.git")
        return False
    
    try:
        import torch
        print("✓ PyTorch:", torch.__version__)
        print("  CUDA available:", torch.cuda.is_available())
    except ImportError:
        print("✗ PyTorch not found")
        return False
    
    try:
        import pandas
        print("✓ pandas:", pandas.__version__)
    except ImportError:
        print("✗ pandas not found")
        return False
    
    return True


def test_improved_detector():
    """Test ImprovedNoveltyDetector"""
    print("\n" + "="*60)
    print("TEST 2: Improved Novelty Detector")
    print("="*60)
    
    try:
        from improved_novelty_detector import ImprovedNoveltyDetector, ThresholdOptimizer, ScoreCalibrator
        print("✓ ImprovedNoveltyDetector imported")
    except ImportError as e:
        print(f"✗ Failed to import ImprovedNoveltyDetector: {e}")
        return False
    
    # Create dummy data
    np.random.seed(42)
    X_train = np.random.randn(200, 512).astype(np.float32)
    X_val = np.random.randn(50, 512).astype(np.float32)
    y_val = np.array([0]*25 + [1]*25)  # Half seen, half novel
    X_test = np.random.randn(100, 512).astype(np.float32)
    y_test = np.array([0]*50 + [1]*50)
    
    train_labels = np.random.randint(0, 3, 200)
    val_labels = np.random.randint(0, 5, 50)
    test_labels = np.random.randint(0, 5, 100)
    class_names = ['class_a', 'class_b', 'class_c']
    
    # Test 1: Basic detector without extras
    print("\nTest 2a: Basic detector (LOF only, no tuning)")
    detector = ImprovedNoveltyDetector(
        n_components=30,
        contamination=0.1,
        use_threshold_tuning=False,
        calibration_method=None,
        ensemble_methods=['lof'],
        use_text_embeddings=False
    )
    
    detector.fit(X_train)
    y_pred = detector.predict(X_test)
    y_scores = detector.predict_proba(X_test)
    
    print(f"  Predictions shape: {y_pred.shape} (expected: (100,))")
    print(f"  Scores shape: {y_scores.shape} (expected: (100,))")
    print(f"  Novel predictions: {np.sum(y_pred == 1)}/100")
    assert y_pred.shape == (100,), "Wrong prediction shape"
    print("  ✓ Basic detector works")
    
    # Test 2: With threshold tuning
    print("\nTest 2b: With threshold tuning")
    detector = ImprovedNoveltyDetector(
        n_components=30,
        contamination=0.1,
        use_threshold_tuning=True,
        calibration_method=None,
        ensemble_methods=['lof'],
        use_text_embeddings=False
    )
    
    detector.fit(X_train, X_val, y_val)
    print(f"  Optimal threshold: {detector.optimal_threshold:.4f}")
    assert detector.optimal_threshold is not None, "Threshold not set"
    print("  ✓ Threshold tuning works")
    
    # Test 3: With calibration
    print("\nTest 2c: With calibration")
    detector = ImprovedNoveltyDetector(
        n_components=30,
        contamination=0.1,
        use_threshold_tuning=True,
        calibration_method='isotonic',
        ensemble_methods=['lof'],
        use_text_embeddings=False
    )
    
    detector.fit(X_train, X_val, y_val)
    y_scores = detector.predict_proba(X_test)
    print(f"  Calibrated scores range: [{y_scores.min():.3f}, {y_scores.max():.3f}]")
    assert detector.calibrator.calibrator is not None, "Calibrator not fitted"
    print("  ✓ Calibration works")
    
    # Test 4: Ensemble methods
    print("\nTest 2d: Ensemble methods")
    for methods in [['ocsvm'], ['iforest'], ['mahalanobis'], ['lof', 'mahalanobis']]:
        detector = ImprovedNoveltyDetector(
            n_components=30,
            contamination=0.1,
            ensemble_methods=methods,
            use_threshold_tuning=False,
            use_text_embeddings=False
        )
        detector.fit(X_train)
        y_pred = detector.predict(X_test)
        print(f"  {'+'.join(methods)}: {np.sum(y_pred == 1)} novel predictions")
    print("  ✓ All ensemble methods work")
    
    # Test 5: With text embeddings
    print("\nTest 2e: With text embeddings")
    try:
        detector = ImprovedNoveltyDetector(
            n_components=30,
            contamination=0.1,
            use_text_embeddings=True,
            ensemble_methods=['lof']
        )
        
        detector.fit(X_train, class_labels=train_labels, class_names=class_names)
        y_pred = detector.predict(X_test, class_labels=test_labels)
        print(f"  Predictions with text: {np.sum(y_pred == 1)} novel")
        print("  ✓ Text embeddings work")
    except Exception as e:
        print(f"  ✗ Text embeddings failed: {e}")
        return False
    
    return True


def test_threshold_optimizer():
    """Test ThresholdOptimizer"""
    print("\n" + "="*60)
    print("TEST 3: Threshold Optimizer")
    print("="*60)
    
    try:
        from improved_novelty_detector import ThresholdOptimizer
        
        # Create dummy validation data
        np.random.seed(42)
        y_true = np.array([0]*50 + [1]*50)
        y_scores = np.random.randn(100)
        y_scores[50:] += 0.5  # Make novel scores slightly higher
        
        # Find optimal threshold
        threshold, metrics = ThresholdOptimizer.find_optimal_threshold(
            y_true, y_scores, metric='hmean'
        )
        
        print(f"  Optimal threshold: {threshold:.4f}")
        print(f"  Seen accuracy: {metrics['seen_accuracy']:.4f}")
        print(f"  Novel accuracy: {metrics['novel_accuracy']:.4f}")
        print(f"  Harmonic mean: {metrics['harmonic_mean']:.4f}")
        
        assert threshold is not None, "No threshold found"
        assert 0 <= metrics['harmonic_mean'] <= 1, "Invalid harmonic mean"
        
        print("  ✓ Threshold optimizer works")
        return True
        
    except Exception as e:
        print(f"  ✗ Threshold optimizer failed: {e}")
        return False


def test_calibrator():
    """Test ScoreCalibrator"""
    print("\n" + "="*60)
    print("TEST 4: Score Calibrator")
    print("="*60)
    
    try:
        from improved_novelty_detector import ScoreCalibrator
        
        np.random.seed(42)
        y_scores = np.random.randn(100)
        y_true = (y_scores > 0).astype(int)
        
        # Test isotonic calibration
        print("\nTest 4a: Isotonic calibration")
        calibrator = ScoreCalibrator(method='isotonic')
        calibrator.fit(y_scores, y_true)
        calibrated = calibrator.transform(y_scores)
        print(f"  Original range: [{y_scores.min():.3f}, {y_scores.max():.3f}]")
        print(f"  Calibrated range: [{calibrated.min():.3f}, {calibrated.max():.3f}]")
        assert calibrator.calibrator is not None, "Calibrator not fitted"
        print("  ✓ Isotonic calibration works")
        
        # Test Platt scaling
        print("\nTest 4b: Platt scaling")
        calibrator = ScoreCalibrator(method='platt')
        calibrator.fit(y_scores, y_true)
        calibrated = calibrator.transform(y_scores)
        print(f"  Calibrated range: [{calibrated.min():.3f}, {calibrated.max():.3f}]")
        print("  ✓ Platt calibration works")
        
        return True
        
    except Exception as e:
        print(f"  ✗ Calibrator failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_ablation_study():
    """Test ablation study functions"""
    print("\n" + "="*60)
    print("TEST 5: Ablation Study")
    print("="*60)
    
    try:
        from ablation_study import quick_comparison
        print("  ✓ Ablation study module imported")
        print("  Note: Full test requires real dataset")
        return True
    except ImportError as e:
        print(f"  ✗ Failed to import ablation_study: {e}")
        return False


def test_multimodal_detector():
    """Test multimodal detector (optional)"""
    print("\n" + "="*60)
    print("TEST 6: Multimodal Detector (Optional)")
    print("="*60)
    
    try:
        from multimodal_novelty_detector import MultimodalPCALOFDetector, MultimodalFeatureExtractor
        print("  ✓ Multimodal detector imported")
        
        # Test feature extractor
        extractor = MultimodalFeatureExtractor()
        class_names = ['bird', 'dog', 'cat']
        text_embeddings = extractor.extract_text_embeddings(class_names)
        print(f"  Text embeddings shape: {text_embeddings.shape}")
        assert text_embeddings.shape == (3, 512), "Wrong text embedding shape"
        print("  ✓ Text extraction works")
        
        return True
        
    except Exception as e:
        print(f"  Note: Multimodal detector not available (optional): {e}")
        return True  # Not critical


def run_all_tests():
    """Run all tests"""
    print("\n" + "="*80)
    print("TESTING ALL IMPROVEMENTS")
    print("="*80)
    
    results = {
        'Imports': test_imports(),
        'Improved Detector': test_improved_detector(),
        'Threshold Optimizer': test_threshold_optimizer(),
        'Score Calibrator': test_calibrator(),
        'Ablation Study': test_ablation_study(),
        'Multimodal Detector': test_multimodal_detector()
    }
    
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    
    for name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{name:<25} {status}")
    
    all_passed = all(results.values())
    
    if all_passed:
        print("\n✓ ALL TESTS PASSED - Ready to run experiments!")
        print("\nNext steps:")
        print("1. Update your dataset_adapter.py to include class names")
        print("2. Run quick comparison: python ablation_study.py --dataset CUB --mode quick")
        print("3. Run full ablation: python ablation_study.py --dataset CUB --mode full")
    else:
        print("\n✗ SOME TESTS FAILED - Please fix issues before proceeding")
        failed = [name for name, passed in results.items() if not passed]
        print(f"Failed tests: {', '.join(failed)}")
    
    return all_passed


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
