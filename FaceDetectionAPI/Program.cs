using FaceDetectionAPI.Services;

var builder = WebApplication.CreateBuilder(args);

// Add services to the container.

builder.Services.AddControllers();
// Learn more about configuring Swagger/OpenAPI at https://aka.ms/aspnetcore/swashbuckle
builder.Services.AddEndpointsApiExplorer();
builder.Services.AddSwaggerGen();

// Register FaceDetectionService with the path to the Haar Cascade file
var haarcascadePath = Path.Combine(builder.Environment.ContentRootPath, "Data", "haarcascade_frontalface_default.xml");
var haarcascadeAltTreePath = Path.Combine(builder.Environment.ContentRootPath, "Data", "haarcascade_frontalface_alt_tree.xml");

builder.Services.AddSingleton(new FaceService(haarcascadePath));

builder.Services.AddSingleton(new SvmService(haarcascadePath));

builder.Services.AddSingleton(new HelperService(haarcascadePath));

var app = builder.Build();

// Configure the HTTP request pipeline.
if (app.Environment.IsDevelopment())
{
    app.UseSwagger();
    app.UseSwaggerUI();
}

app.UseHttpsRedirection();

app.UseAuthorization();

app.MapControllers();

app.Run();
