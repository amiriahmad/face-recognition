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

    private readonly Service3 _service3;

    public FaceDetectionController(FaceService faceDetectionService
        //EigenService eigenService,
        //Service3 service3
        )
    {
        _faceService = faceDetectionService;
        //_eigenService = eigenService;
        //_service3 = service3;
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

    //[HttpPost("EigenMatch")]
    //public IActionResult EigenMatch()
    //{
    //    var res = _eigenService.NormalizeAndRecognizeFace();
    //    //var res = _eigenService.DetectAndRecognizeFace();

    //    if (res?.Count > 0)
    //    {
    //        return Ok(res);
    //    }

    //    return BadRequest("hhhmmmmmmmmmmmmmmmmmmmm");


    //}





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
