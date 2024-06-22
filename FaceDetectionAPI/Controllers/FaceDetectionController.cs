using FaceDetectionAPI.Services;

using Microsoft.AspNetCore.Http;
using Microsoft.AspNetCore.Mvc;

namespace FaceDetectionAPI.Controllers;
[Route("api/[controller]")]
[ApiController]
public class FaceDetectionController : ControllerBase
{
    private readonly FaceService _faceService;
    private readonly EigenService _eigenService;

    private readonly SvmService _svmService;

    public FaceDetectionController(
        FaceService faceDetectionService,
        SvmService svmService
        )
    {
        _faceService = faceDetectionService;
        _svmService = svmService;
    }



    [HttpPost("DetectFaces")]
    public IActionResult DetectFaces()
    {
        var faces = _faceService.DetectFaces();

        return Ok(faces);
    }
    [HttpPost("Match")]
    public IActionResult TestMatch()
    {
        var res = _faceService.RecognizeFace();
        return Ok(res);

    }

    [HttpPost("SvmRecognizer")]
    public IActionResult SvmRecognizer()
    {

        var result = _svmService.RecognizeFace();

        return Ok(result);

    }





    //[HttpPost("detect")]
    //public IActionResult DetectFaces([FromForm] IFormFile image)
    //{
    //    if (image == null || image.Length == 0)
    //    {
    //        return BadRequest("No image uploaded.");
    //    }

    //    using var ms = new MemoryStream();
    //    image.CopyTo(ms);
    //    var imageBytes = ms.ToArray();

    //    var faces = _faceService.DetectFaces(imageBytes);

    //    return Ok(faces);
    //}



    //[HttpPost("Match2")]
    //public IActionResult Match2()
    //{
    //    var datasetPath = @"D:\Projects\Demos\FaceDetectionAPI\FaceDetectionAPI\Dataset\";
    //    var imagePaths = new List<string>();
    //    var labels = new List<int>();

    //    var label = 0;
    //    foreach (var dir in Directory.GetDirectories(datasetPath))
    //    {
    //        foreach (var file in Directory.GetFiles(dir, "*.jpeg"))
    //        {
    //            imagePaths.Add(file);
    //            labels.Add(label);
    //        }
    //        label++;
    //    }


    //    int k = 5; // 5-fold cross-validation

    //    var (bestRadius, bestNeighbors, bestGridX, bestGridY) = _face2Service.GridSearch(imagePaths, labels, k);

    //    return Ok(new { R = bestRadius, N = bestNeighbors, X = bestGridX, Y = bestGridY });
    //}
}
