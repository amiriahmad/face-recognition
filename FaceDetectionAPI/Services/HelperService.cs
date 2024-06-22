using System.Drawing;
using System.Drawing.Imaging;
using System.Runtime.InteropServices;

using Emgu.CV;
using Emgu.CV.CvEnum;
using Emgu.CV.Structure;


namespace FaceDetectionAPI.Services;

public class HelperService
{
    private readonly string _datasetPath = @"D:\Projects\Personal\FaceDetectionAPI\FaceDetectionAPI\Dataset\TrainingModels\";
    public const int _imageWidth = 100;
    public const int _imageHeight = 100;
    private readonly string _haarcascadePath;
    private readonly CascadeClassifier _faceDetector;
    private int _neighbors = 13;

    public HelperService(string haarcascadePath)
    {
        _haarcascadePath = haarcascadePath;
        _faceDetector = new CascadeClassifier(haarcascadePath);
    }



    public List<Image<Gray, byte>> LoadImages(out List<int> labels)
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
                    var face = grayImages[i].Resize(_imageWidth, _imageHeight, Inter.LinearExact);
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
                    var face = grayImages[i].Resize(_imageWidth, _imageHeight, Inter.LinearExact);
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

    public Image<Bgr, byte> ConvertBitmapToImage(Bitmap bitmap)
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

    public List<Image<Gray, byte>> DetectAndNormalizeFace(Image<Gray, byte> image)
    {
        var faces = _faceDetector.DetectMultiScale(image, 1.1, _neighbors, new System.Drawing.Size(20, 20));
        //var faces = _faceDetector.DetectMultiScale(image);

        List<Image<Gray, byte>> images = [];

        if (faces.Length > 0)
        {
            //var face = faces[0];
            foreach (var face in faces)
            {
                var faceImg = image.GetSubRect(face).Resize(_imageWidth, _imageHeight, Inter.LinearExact);
                CvInvoke.EqualizeHist(faceImg, faceImg);
                images.Add(faceImg);

            }

            //var faceImg = image.GetSubRect(face).Resize(200, 200, Inter.Cubic);
            //CvInvoke.EqualizeHist(faceImg, faceImg);
            //return faceImg;
        }

        return images;


    }

    public List<Image<Gray, byte>>? AugmentData(Image<Gray, byte> image)
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

        return augmentedImages.Count > 0 ? augmentedImages : null;
    }


}
