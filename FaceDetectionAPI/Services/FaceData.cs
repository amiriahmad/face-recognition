using Emgu.CV;
using Emgu.CV.Structure;

namespace FaceDetectionAPI.Services;

// this is a test change , ok

public class FaceData
{
    public string PersonName { get; set; }
    public Image<Gray, byte> FaceImage { get; set; }
    public DateTime CreateDate { get; set; }
}
