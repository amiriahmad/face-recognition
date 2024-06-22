using System.Drawing;

using Emgu.CV;
using Emgu.CV.CvEnum;
using Emgu.CV.Face;
using Emgu.CV.ML;
using Emgu.CV.Structure;
using Emgu.CV.Util;

namespace FaceDetectionAPI.Services;

public class SvmService
{
    private readonly LBPHFaceRecognizer _faceRecognizer;
    private readonly SVM _svm;
    private List<string> _imgPaths = [];
    private readonly int _imageWidth = 100;
    private readonly int _imageHeight = 100;
    private readonly double _svmThreshold = 0.0;
    private CascadeClassifier _haarCascade;
    private readonly string _datasetPath = @"D:\Projects\Personal\FaceDetectionAPI\FaceDetectionAPI\Dataset\TrainingModels\";

    public SvmService(string haarcascadePath)
    {
        _haarCascade = new CascadeClassifier(haarcascadePath);
        _faceRecognizer = new LBPHFaceRecognizer();
        _svm = new SVM
        {
            C = 1
        };
        TrainModel();
    }



    private List<Image<Gray, byte>> LoadImages(out List<int> labels)
    {
        var images = new List<Image<Gray, byte>>();
        labels = new List<int>();
        var label = 0;

        foreach (var dir in Directory.GetDirectories(_datasetPath))
        {
            foreach (var file in Directory.GetFiles(dir, "*.jpg"))
            {
                var img = new Image<Gray, byte>(file).Resize(_imageWidth, _imageHeight, Inter.Linear);
                images.Add(img);
                labels.Add(label);
            }
            label++;
        }
        return images;
    }

    private Mat ExtractLBPHFeatures(List<Image<Gray, byte>> images)
    {
        var lbphHistograms = new List<float[]>();

        foreach (var image in images)
        {
            var lbph = ComputeLBPH(image);
            lbphHistograms.Add(lbph);
        }

        // Convert List<float[]> to Mat
        var numFeatures = lbphHistograms[0].Length;
        var lbphFeatures = new Mat(lbphHistograms.Count, numFeatures, DepthType.Cv32F, 1);

        //for (int i = 0; i < lbphHistograms.Count; i++)
        //{
        //    //var featureRow = new Mat(1, numFeatures, DepthType.Cv32F, 1);
        //    //featureRow.SetTo(lbphHistograms[i]);
        //    //lbphFeatures.SetTo(featureRow);


        //    lbphFeatures.SetTo<float>(lbphHistograms[i]);
        //}

        for (int i = 0; i < lbphHistograms.Count; i++)
        {
            for (int j = 0; j < numFeatures; j++)
            {
                unsafe
                {
                    // Access the pointer to the element at (i, j) in the Mat
                    float* ptr = (float*)lbphFeatures.DataPointer + i * lbphFeatures.Cols + j;

                    // Set the value using pointer arithmetic
                    *ptr = lbphHistograms[i][j];
                }
            }
        }

        return lbphFeatures;
    }

    private void TrainModel()
    {
        var images = LoadImages(out var labels);
        var iInputArrayOfArrays = new VectorOfMat(images.Select(image => image.Mat).ToArray());
        var iLabels = new VectorOfInt(labels.ToArray());
        //_faceRecognizer.Train(iInputArrayOfArrays, iLabels);

        // Extract LBPH features
        var lbphFeatures = ExtractLBPHFeatures(images);

        // Convert labels to Mat
        var labelsMat = new Mat(labels.Count, 1, Emgu.CV.CvEnum.DepthType.Cv32S, 1);
        labelsMat.SetTo<int>(labels.ToArray());

        // Train SVM classifier
        _svm.Train(lbphFeatures, Emgu.CV.ML.MlEnum.DataLayoutType.RowSample, labelsMat);
    }

    public float RecognizeFace()
    {
        //string imagePath = @"D:\Projects\Personal\FaceDetectionAPI\FaceDetectionAPI\Dataset\TrainingModels\Y\y2.jpg";
        string imagePath = @"D:\Projects\Personal\FaceDetectionAPI\FaceDetectionAPI\Dataset\TestingModels\y5.jpg";

        using (var grayImage = new Image<Gray, byte>(imagePath))
        {
            var resizedImage = grayImage.Resize(_imageWidth, _imageHeight, Emgu.CV.CvEnum.Inter.Linear);

            // Extract LBPH features
            var lbph = ComputeLBPH(resizedImage);

            // Convert features to Mat
            var lbphMat = new Mat(1, lbph.Length, Emgu.CV.CvEnum.DepthType.Cv32F, 1);
            lbphMat.SetTo(lbph);

            // Predict with SVM
            var response = _svm.Predict(lbphMat);
            return response;
        }
    }


    private static float[] ComputeLBPH(Image<Gray, byte> image)
    {
        int radius = 1;
        int neighbors = 8;
        int gridX = 8;
        int gridY = 8;
        var lbpImage = new Image<Gray, byte>(image.Size);

        // Compute LBP for each pixel
        for (int i = 0; i < image.Rows; i++)
        {
            for (int j = 0; j < image.Cols; j++)
            {
                var center = image.Data[i, j, 0];
                byte code = 0;
                for (int k = 0; k < neighbors; k++)
                {
                    int x = j + (int)(radius * Math.Cos(2.0 * Math.PI * k / neighbors));
                    int y = i - (int)(radius * Math.Sin(2.0 * Math.PI * k / neighbors));
                    if (x >= 0 && x < image.Cols && y >= 0 && y < image.Rows)
                    {
                        byte pixel = image.Data[y, x, 0];
                        code |= (byte)((pixel > center ? 1 : 0) << k);
                    }
                }
                lbpImage.Data[i, j, 0] = code;
            }
        }

        // Calculate histograms for each grid cell
        var histograms = new List<float>();
        int cellWidth = image.Width / gridX;
        int cellHeight = image.Height / gridY;
        for (int gx = 0; gx < gridX; gx++)
        {
            for (int gy = 0; gy < gridY; gy++)
            {
                var cell = new Rectangle(gx * cellWidth, gy * cellHeight, cellWidth, cellHeight);
                var cellHist = new Mat();
                var cellMat = new Mat(lbpImage.Mat, cell);
                var mask = new Mat(); // No mask

                CvInvoke.CalcHist(
                            new VectorOfMat([cellMat]),       // Source images
                             [0],             // Channels
                             mask,                        // Optional mask
                             cellHist,                    // Output histogram
                             [256],           // Histogram size
                             [0, 256],      // Ranges
                             true
                         );

                var cellHistValues = new float[256];
                cellHist.CopyTo(cellHistValues);
                histograms.AddRange(cellHistValues);
            }
        }

        return histograms.ToArray();
    }



}






