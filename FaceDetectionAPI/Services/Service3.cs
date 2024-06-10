using System.Drawing;

using Emgu.CV;
using Emgu.CV.CvEnum;
using Emgu.CV.Face;
using Emgu.CV.Structure;
using Emgu.CV.Util;

namespace FaceDetectionAPI.Services;

public class Service3
{
    private CascadeClassifier haarCascade;
    private Image<Bgr, byte> bgrFrame = null;
    private Image<Gray, byte> detectedFace = null;
    private List<FaceData> faceList = [];
    private VectorOfMat imageList = new();
    private List<string> nameList = [];
    private VectorOfInt labelList = new();

    private string _facePhotosPath;
    private string _faceListTextFile;

    private readonly LBPHFaceRecognizer recognizer;
    private readonly IWebHostEnvironment _environment;

    public string FaceName { get; set; }

    public Bitmap CameraCaptureFace { get; set; }
    public Bitmap CameraCapture { get; set; }

    public Service3(IWebHostEnvironment environment, string haarcascadePath)
    {
        recognizer = new();
        haarCascade = new CascadeClassifier(haarcascadePath);
        _environment = environment;
        CreateFilesAndDirs();

        //AddNewFace();

    }


    public void AddNewFace()
    {
        try
        {
            string imagePath = @"D:\Projects\Demos\FaceDetectionAPI\FaceDetectionAPI\Dataset\Me\a1.jpg";
            detectedFace = new Image<Gray, byte>(imagePath);

            if (detectedFace == null)
            {
                return;
            }
            //Save detected face
            detectedFace = detectedFace.Resize(100, 100, Inter.Cubic);
            var savingPath = $@"{_facePhotosPath}\face{faceList.Count + 1}.bmp";
            detectedFace.Save(savingPath);

            using var writer = new StreamWriter(_faceListTextFile, true);
            writer.WriteLine(string.Format("face{0}:{1}", (faceList.Count + 1), $"person{faceList.Count + 1}"));
        }
        catch (Exception)
        {

            throw;
        }

        //GetFacesList();

    }


    private void ProcessFrame()
    {
        string imagePath = @"D:\Projects\Demos\FaceDetectionAPI\FaceDetectionAPI\Dataset\Test\a2.jpg";
        bgrFrame = new Image<Bgr, byte>(imagePath); // videoCapture.QueryFrame().ToImage<Bgr, Byte>();

        if (bgrFrame != null)
        {
            try
            {//for emgu cv bug
                Image<Gray, byte> grayframe = bgrFrame.Convert<Gray, byte>();

                Rectangle[] faces = haarCascade.DetectMultiScale(grayframe, 1.2, 10, new System.Drawing.Size(50, 50), new System.Drawing.Size(200, 200));

                //detect face
                FaceName = "No face detected";
                foreach (var face in faces)
                {
                    bgrFrame.Draw(face, new Bgr(255, 255, 0), 2);
                    detectedFace = bgrFrame.Copy(face).Convert<Gray, byte>();
                    FaceRecognition();
                    break;
                }
                CameraCapture = bgrFrame.ToBitmap();
            }
            catch (Exception ex)
            {

                //todo log
            }

        }
    }

    public void GetFacesList()
    {

        faceList.Clear();
        string line;
        FaceData? faceInstance = null;



        using var reader = new StreamReader(_faceListTextFile);
        int i = 0;
        while ((line = reader!.ReadLine()) is not null)
        {
            string[] lineParts = line.Split(':');
            faceInstance = new FaceData();
            faceInstance.FaceImage = new Image<Gray, byte>(_facePhotosPath + lineParts[0] + ".bmp");
            faceInstance.PersonName = lineParts[1];
            faceList.Add(faceInstance);
        }
        foreach (var face in faceList)
        {
            imageList.Push(face.FaceImage.Mat);
            nameList.Add(face.PersonName);
            labelList.Push(new[] { i++ });
        }

        // Train recogniser
        if (imageList.Size > 0)
        {
            //recognizer = new LBPHFaceRecognizer();
            //recognizer = new EigenFaceRecognizer(imageList.Size);
            recognizer.Train(imageList, labelList);
        }

    }

    private void FaceRecognition()
    {
        if (imageList.Size != 0)
        {
            //Eigen Face Algorithm
            FaceRecognizer.PredictionResult result = recognizer.Predict(detectedFace.Resize(100, 100, Inter.Cubic));
            FaceName = nameList[result.Label];
            CameraCaptureFace = detectedFace.ToBitmap();
        }
        else
        {
            FaceName = "Please Add Face";
        }
    }

    private void CreateFilesAndDirs()
    {
        if (!Directory.Exists(Path.Combine(_environment.ContentRootPath, "dataset", "faces")))
        {
            Directory.CreateDirectory(Path.Combine(_environment.ContentRootPath, "dataset", "faces"));
        }

        if (!File.Exists(Path.Combine(_environment.ContentRootPath, "Data", "FaceLabelList.txt")))
        {
            File.Create(Path.Combine(_environment.ContentRootPath, "Data", "FaceLabelList.txt"))
                .Close();
        }

        _facePhotosPath = Path.Combine(_environment.ContentRootPath, "dataset", "faces");
        _faceListTextFile = Path.Combine(_environment.ContentRootPath, "Data", "FaceLabelList.txt");
    }


}
