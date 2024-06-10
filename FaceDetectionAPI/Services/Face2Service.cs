using Emgu.CV;
using Emgu.CV.CvEnum;
using Emgu.CV.Face;
using Emgu.CV.Structure;
using Emgu.CV.Util;

namespace FaceDetectionAPI.Services;

public class Face2Service
{

    private List<string> _imgPaths = [];
    List<Image<Bgr, byte>> _bgrImages = [];
    List<Image<Gray, byte>> _grayImages = [];
    private readonly string _haarcascadePath;
    private LBPHFaceRecognizer _recognizer;
    private readonly CascadeClassifier _faceDetector;
    private double _threshold = 0.6;
    private string datasetPath = @"D:\Projects\Demos\FaceDetectionAPI\FaceDetectionAPI\Dataset\";

    public Face2Service(string haarcascadePath)
    {
        _haarcascadePath = haarcascadePath;
        _recognizer = new LBPHFaceRecognizer();
        _faceDetector = new CascadeClassifier(haarcascadePath);

        foreach (var dir in Directory.GetDirectories(datasetPath))
        {
            foreach (var file in Directory.GetFiles(dir, "*.jpeg"))
            {
                _imgPaths.Add(file);
            }
        }


    }


    public LBPHFaceRecognizer TrainRecognizer(List<string> imagePaths, List<int> labels, int radius, int neighbors, int gridX, int gridY)
    {
        var recognizer = new LBPHFaceRecognizer(radius, neighbors, gridX, gridY, 100);

        var images = new List<Image<Gray, byte>>();
        var augmentedLabels = new List<int>();

        for (int i = 0; i < imagePaths.Count; i++)
        {
            var img = new Image<Gray, byte>(imagePaths[i]);
            var face = DetectAndNormalizeFace(img);
            if (face != null)
            {
                var augmentedImages = AugmentData(face);
                images.AddRange(augmentedImages);
                augmentedLabels.AddRange(Enumerable.Repeat(labels[i], augmentedImages.Count));
            }
        }

        if (images.Count != augmentedLabels.Count)
        {
            throw new InvalidOperationException($"The number of samples (images) must equal the number of labels. Was len(samples)={images.Count}, len(labels)={augmentedLabels.Count}.");
        }

        var iInputArrayOfArrays = new VectorOfMat(images.Select(image => image.Mat).ToArray());
        var iLabels = new VectorOfInt(augmentedLabels.ToArray());

        recognizer.Train(iInputArrayOfArrays, iLabels);

        return recognizer;
    }
    public double CrossValidate(List<string> imagePaths, List<int> labels, int k, int radius, int neighbors, int gridX, int gridY)
    {
        int n = imagePaths.Count;
        int foldSize = n / k;

        var accuracies = new List<double>();

        for (int i = 0; i < k; i++)
        {
            var testImagePaths = new List<string>();
            var testLabels = new List<int>();
            var trainImagePaths = new List<string>();
            var trainLabels = new List<int>();

            for (int j = 0; j < n; j++)
            {
                if (j >= i * foldSize && j < (i + 1) * foldSize)
                {
                    testImagePaths.Add(imagePaths[j]);
                    testLabels.Add(labels[j]);
                }
                else
                {
                    trainImagePaths.Add(imagePaths[j]);
                    trainLabels.Add(labels[j]);
                }
            }

            _recognizer = TrainRecognizer(trainImagePaths, trainLabels, radius, neighbors, gridX, gridY);

            int correct = 0;
            for (int j = 0; j < testImagePaths.Count; j++)
            {
                var img = new Image<Gray, byte>(testImagePaths[j]);
                var face = DetectAndNormalizeFace(img);

                if (RecognizeFace(face, out int predictedLabel, out double confidence))
                {
                    if (predictedLabel == testLabels[j])
                    {
                        correct++;
                    }
                }
            }

            double accuracy = (double)correct / testImagePaths.Count;
            accuracies.Add(accuracy);
        }

        return accuracies.Average();
    }
    public (int, int, int, int) GridSearch(List<string> imagePaths, List<int> labels, int k)
    {
        var bestParams = (radius: 1, neighbors: 8, gridX: 8, gridY: 8);
        double bestAccuracy = 0;

        var radii = new[] { 1, 2, 3 };
        var neighbors = new[] { 8, 16, 24 };
        var gridX = new[] { 8, 10, 12 };
        var gridY = new[] { 8, 10, 12 };

        foreach (var radius in radii)
        {
            foreach (var neighbor in neighbors)
            {
                foreach (var gx in gridX)
                {
                    foreach (var gy in gridY)
                    {
                        double accuracy = CrossValidate(imagePaths, labels, k, radius, neighbor, gx, gy);
                        if (accuracy > bestAccuracy)
                        {
                            bestAccuracy = accuracy;
                            bestParams = (radius, neighbor, gx, gy);
                        }

                        Console.WriteLine($"Params: radius={radius}, neighbors={neighbor}, gridX={gx}, gridY={gy}, Accuracy={accuracy}");
                    }
                }
            }
        }

        return bestParams;
    }
    public bool RecognizeFace(Image<Gray, byte> face, out int label, out double confidence)
    {
        if (face != null)
        {
            var predictionResult = _recognizer.Predict(face);
            label = predictionResult.Label;
            confidence = predictionResult.Distance;
            return true;
        }

        label = -1;
        confidence = double.MaxValue;
        return false;
    }


    private Image<Gray, byte> DetectAndNormalizeFace(Image<Gray, byte> image)
    {
        var faces = _faceDetector.DetectMultiScale(image, 1.1, 10, new System.Drawing.Size(20, 20));

        if (faces.Length > 0)
        {
            var face = faces[0];
            var faceImg = image.GetSubRect(face).Resize(100, 100, Inter.Cubic);
            CvInvoke.EqualizeHist(faceImg, faceImg);
            return faceImg;
        }

        return null;
    }

    private List<Image<Gray, byte>> AugmentData(Image<Gray, byte> image)
    {
        var augmentedImages = new List<Image<Gray, byte>>();
        var scales = new double[] { 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.4, 1.6 };

        foreach (var scale in scales)
        {
            var resized = image.Resize(scale, Inter.Linear);
            var normalized = DetectAndNormalizeFace(resized);
            if (normalized != null)
            {
                augmentedImages.Add(normalized);
            }
        }

        return augmentedImages;
    }




}
