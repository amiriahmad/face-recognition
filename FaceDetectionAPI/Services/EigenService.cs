using System.Drawing;
using System.Reflection.Emit;

using Emgu.CV;
using Emgu.CV.CvEnum;
using Emgu.CV.Face;
using Emgu.CV.Structure;
using Emgu.CV.Util;

namespace FaceDetectionAPI.Services;

public class EigenService
{
    private CascadeClassifier _faceCascade;
    private LBPHFaceRecognizer _LBPHFaceRecognizer;
    private EigenFaceRecognizer _eigenFaceRecognizer;
    private List<string> _imgPaths = [];
    private List<Image<Gray, byte>> _trainingImages = [];
    private List<int> _labels = [];
    private double _threshold = 0.6;
    private string datasetPath = @"D:\Projects\Demos\FaceDetectionAPI\FaceDetectionAPI\Dataset\Me\";

    public EigenService(string haarcascadePath)
    {
        _eigenFaceRecognizer = new EigenFaceRecognizer();
        _LBPHFaceRecognizer = new();
        _faceCascade = new CascadeClassifier(haarcascadePath);

        LoadTrainingData();
    }

    private void LoadTrainingData()
    {
        string[] labelData = File.ReadAllLines(@"D:\Projects\Demos\FaceDetectionAPI\FaceDetectionAPI\Data\labels.txt");

        foreach (var line in labelData)
        {
            var parts = line.Split(',');
            var imagePath = Path.Combine(datasetPath, parts[0]);
            var label = int.Parse(parts[1]);

            var img = new Image<Gray, byte>(imagePath);
            _trainingImages.Add(img.Resize(100, 100, Inter.Cubic));
            _labels.Add(label);
        }

        if (_trainingImages.Count > 0)
        {
            var iInputArrayOfArrays = new VectorOfMat(_trainingImages.Select(image => image.Mat).ToArray());
            var iLabels = new VectorOfInt(_labels.ToArray());
            _eigenFaceRecognizer.Train(iInputArrayOfArrays, iLabels);
            //_LBPHFaceRecognizer.Train(iInputArrayOfArrays, iLabels);
        }
    }

    public List<double> DetectAndRecognizeFace()
    {
        List<double> confidences = [];

        //string imagePath = @"D:\Projects\Demos\FaceDetectionAPI\FaceDetectionAPI\Dataset\Test\k4.jpg";
        string imagePath = @"D:\Projects\Demos\FaceDetectionAPI\FaceDetectionAPI\Dataset\Test\a2.jpg";
        // Convert to grayscale
        var grayImage = new Image<Gray, byte>(imagePath);

        // Detect faces
        var faces = _faceCascade.DetectMultiScale(grayImage, 1.1, 10, Size.Empty);

        foreach (var face in faces)
        {
            // Recognize face
            var faceImage = grayImage.Copy(face).Resize(100, 100, Inter.Cubic);
            var result = _LBPHFaceRecognizer.Predict(faceImage);
            //var result = _eigenFaceRecognizer.Predict(faceImage);

            if (result.Label != -1)
            {
                confidences.Add(result.Distance);
            }

        }


        return confidences;


    }

    public List<double> NormalizeAndRecognizeFace()
    {
        List<double> confidences = [];

        string imagePath = @"D:\Projects\Demos\FaceDetectionAPI\FaceDetectionAPI\Dataset\Test\k4.jpg";
        //string imagePath = @"D:\Projects\Demos\FaceDetectionAPI\FaceDetectionAPI\Dataset\Test\a2.jpg";
        // Convert to grayscale
        var grayImage = new Image<Gray, byte>(imagePath);

        var augmenteds = AugmentData(grayImage);

        foreach (var aug in augmenteds)
        {
            var faces = _faceCascade.DetectMultiScale(aug, 1.1, 10, Size.Empty);
            foreach (var face in faces)
            {
                // Recognize face
                var faceImage = grayImage.Copy(face).Resize(100, 100, Inter.Cubic);
                //var result = _LBPHFaceRecognizer.Predict(faceImage);
                var result = _eigenFaceRecognizer.Predict(faceImage);

                if (result.Label != -1)
                {
                    confidences.Add(result.Distance);
                }

            }

        }

        // Detect faces




        return confidences;


    }

    private List<Image<Gray, byte>> DetectAndNormalizeFace(Image<Gray, byte> image)
    {
        var faces = _faceCascade.DetectMultiScale(image, 1.1, 10, new System.Drawing.Size(20, 20));

        List<Image<Gray, byte>> images = [];

        if (faces.Length > 0)
        {
            //var face = faces[0];
            foreach (var face in faces)
            {
                var faceImg = image.GetSubRect(face).Resize(200, 200, Inter.NearestExact);
                CvInvoke.EqualizeHist(faceImg, faceImg);
                images.Add(faceImg);

            }

            return images;
            //var faceImg = image.GetSubRect(face).Resize(200, 200, Inter.Cubic);
            //CvInvoke.EqualizeHist(faceImg, faceImg);
            //return faceImg;
        }

        return null;
    }

    private List<Image<Gray, byte>> AugmentData(Image<Gray, byte> image)
    {
        var augmentedImages = new List<Image<Gray, byte>>();
        var scales = new double[] { 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0, 1.1, 1.2, 1.4, 1.6, 1.8, 2.0, 2, 2 };

        foreach (var scale in scales)
        {
            var resized = image.Resize(scale, Inter.Cubic);
            var normalized = DetectAndNormalizeFace(resized);
            if (normalized is not null)
            {
                augmentedImages.AddRange(normalized);
            }
        }

        return augmentedImages;
    }
}
