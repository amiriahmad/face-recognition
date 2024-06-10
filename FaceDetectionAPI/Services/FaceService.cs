using System.Drawing;
using System.Drawing.Imaging;
using System.Runtime.InteropServices;

using Emgu.CV;
using Emgu.CV.CvEnum;
using Emgu.CV.Face;
using Emgu.CV.Structure;
using Emgu.CV.Util;

namespace FaceDetectionAPI.Services;

public class FaceService
{
    private List<string> _imgPaths = [];
    List<Image<Bgr, byte>> _bgrImages = [];
    List<Image<Gray, byte>> _grayImages = [];
    private readonly string _haarcascadePath;
    private readonly LBPHFaceRecognizer _recognizer;
    private readonly CascadeClassifier _faceDetector;
    private readonly double _threshold = 100;
    private readonly int _neighbors = 13;
    private readonly string _datasetPath = @"D:\Projects\Personal\FaceDetectionAPI\FaceDetectionAPI\Dataset\TrainingModels\";

    int _faceSize = 400;

    public FaceService(string haarcascadePath)
    {
        _haarcascadePath = haarcascadePath;
        _faceDetector = new CascadeClassifier(haarcascadePath);
        _recognizer = new LBPHFaceRecognizer(1, _neighbors, 8, 8, _threshold);

        LoadTrainingData();

    }

    private void LoadTrainingData()
    {
        foreach (var dir in Directory.GetDirectories(_datasetPath))
        {
            foreach (var file in Directory.GetFiles(dir, "*.jpg"))
            {
                _imgPaths.Add(file);
            }
        }

        TrainRecognizer();
    }

    public int DetectFaces()
    {
        foreach (var item in _imgPaths)
        {
            _bgrImages.Add(new Image<Bgr, byte>(item));
        }

        foreach (var image in _bgrImages)
        {
            _grayImages.Add(image.Convert<Gray, byte>());
        }

        // Initialize the face detector and face recognizer
        var faceDetector = new CascadeClassifier(_haarcascadePath);

        List<Rectangle[]> rectangles = [];
        foreach (var item in _grayImages)
        {
            var faces = faceDetector.DetectMultiScale(item, 1.1, 10, Size.Empty, Size.Empty);
            rectangles.Add(faces);
        }

        return rectangles.Count;
    }

    public Rectangle[] DetectFaces(byte[] imageBytes)
    {

        using var ms = new MemoryStream(imageBytes);
        using var bitmap = new Bitmap(ms);

        // Convert the bitmap to Emgu.CV Image
        var img = ConvertBitmapToImage(bitmap);

        // Convert to grayscale
        var grayImg = img.Convert<Gray, byte>();

        // Load the Haar Cascade
        var faceDetector = new CascadeClassifier(_haarcascadePath);

        // Detect faces
        var faces = faceDetector.DetectMultiScale(grayImg, 1.1, 10, Size.Empty, Size.Empty);

        return faces;
    }

    public bool TrainRecognizer()
    {
        var labels = new List<int>();
        var label = 0;

        try
        {
            foreach (var file in _imgPaths)
            {
                var img = new Image<Gray, byte>(file);
                var augmenteds = AugmentData(img);
                if (augmenteds is not null && augmenteds.Count > 0)
                {
                    _grayImages.AddRange(augmenteds);
                    labels.AddRange(Enumerable.Repeat(label, augmenteds.Count));
                    SaveFace(_grayImages, label);
                }

                label++;
            }
            var iInputArrayOfArrays = new VectorOfMat(_grayImages.Select(image => image.Mat).ToArray());
            var iLabels = new VectorOfInt(labels.ToArray());

            _recognizer.Train(iInputArrayOfArrays, iLabels);
            return true;
        }
        catch (Exception)
        {

            throw;
        }



    }

    public double RecognizeFace()
    {
        //string imagePath = @"D:\Projects\Demos\FaceDetectionAPI\FaceDetectionAPI\Dataset\TestingModels\a2.jpg";
        string imagePath = @"D:\Projects\Personal\FaceDetectionAPI\FaceDetectionAPI\Dataset\TestingModels\ak1.jpg";
        //string imagePath = @"D:\Projects\Demos\FaceDetectionAPI\FaceDetectionAPI\Dataset\TestingModels\k4.jpg";

        double confidence = 0;

        var image = new Image<Gray, byte>(imagePath);

        //var faces = AugmentData(image);
        var faces = AugmentData(image) ?? throw new Exception("no face!!!");

        SaveFace(faces, 33333);

        //var faces = DetectAndNormalizeFace(image) ?? throw new Exception("no face!!!");



        if (faces is not null)
        {
            foreach (var f in faces)
            {
                var predictionResult = _recognizer.Predict(f);
                if (predictionResult.Distance > confidence)
                {
                    confidence = predictionResult.Distance;
                }
            }


        }

        //var predictionResult = _recognizer.Predict(faces[0]);

        //confidence = predictionResult.Distance;

        return confidence; // predictionResult.Distance;//< _threshold;
    }

    private Image<Bgr, byte> ConvertBitmapToImage(Bitmap bitmap)
    {
        var width = bitmap.Width;
        var height = bitmap.Height;

        var image = new Image<Bgr, byte>(width, height);

        var bitmapData = bitmap.LockBits(new Rectangle(0, 0, width, height), ImageLockMode.ReadOnly, bitmap.PixelFormat);

        // Get the address of the first line
        var ptr = bitmapData.Scan0;

        // Declare an array to hold the bytes of the bitmap
        var bytes = new byte[Math.Abs(bitmapData.Stride) * height];

        // Copy the RGB values into the array
        Marshal.Copy(ptr, bytes, 0, bytes.Length);

        // Unlock the bits
        bitmap.UnlockBits(bitmapData);

        // Set pixels of the image
        for (var y = 0; y < height; y++)
        {
            for (var x = 0; x < width; x++)
            {
                var index = y * bitmapData.Stride + x * Image.GetPixelFormatSize(bitmap.PixelFormat) / 8;

                var blue = bytes[index];
                var green = bytes[index + 1];
                var red = bytes[index + 2];

                image[y, x] = new Bgr(blue, green, red);
            }
        }

        return image;
    }

    private List<Image<Gray, byte>> DetectAndNormalizeFace(Image<Gray, byte> image)
    {
        var faces = _faceDetector.DetectMultiScale(image, 1.1, _neighbors, new System.Drawing.Size(20, 20));
        //var faces = _faceDetector.DetectMultiScale(image);

        List<Image<Gray, byte>> images = [];

        if (faces.Length > 0)
        {
            //var face = faces[0];
            foreach (var face in faces)
            {
                var faceImg = image.GetSubRect(face).Resize(_faceSize, _faceSize, Inter.LinearExact);
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
        var scales = new double[] { 0.6, 0.8, 1.0, 1.2, 1.4, 1.6 };

        foreach (var scale in scales)
        {
            var resized = image.Resize(scale, Inter.LinearExact);
            var normalized = DetectAndNormalizeFace(resized);
            if (normalized is not null)
            {
                augmentedImages.AddRange(normalized);
            }
        }

        return augmentedImages;
    }

    public void SaveFace(List<Image<Gray, byte>> grayImages, int label)
    {
        try
        {

            if (grayImages is null || grayImages.Count == 0)
            {
                return;
            }

            //Save detected face

            if (label > 30000)
            {
                for (var i = 0; i < grayImages.Count; i++)
                {
                    var face = grayImages[i].Resize(_faceSize, _faceSize, Inter.LinearExact);
                    var savingPath = $@"D:\Projects\Personal\FaceDetectionAPI\FaceDetectionAPI\DetectedFaces\face{i + 1}.bmp";
                    face.Save(savingPath);

                    //using var writer = new StreamWriter(@"D:\Projects\Personal\FaceDetectionAPI\FaceDetectionAPI\Data\FaceLabelList.txt", true);
                    //writer.WriteLine(string.Format("face{0}:{1}", (i + 1), $"person{label}"));
                }
            }
            else
            {
                for (var i = 0; i < grayImages.Count; i++)
                {
                    var face = grayImages[i].Resize(_faceSize, _faceSize, Inter.LinearExact);
                    var savingPath = $@"D:\Projects\Personal\FaceDetectionAPI\FaceDetectionAPI\Faces\face{i + 1}.bmp";
                    face.Save(savingPath);

                    //using var writer = new StreamWriter(@"D:\Projects\Personal\FaceDetectionAPI\FaceDetectionAPI\Data\FaceLabelList.txt", true);
                    //writer.WriteLine(string.Format("face{0}:{1}", (i + 1), $"person{label}"));
                }
            }

        }
        catch (Exception)
        {

            throw;
        }


    }


}