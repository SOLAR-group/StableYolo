# StableYolo

This project contains the implementation of StableYolo. StableYolo uses evolutionary computation to improve the quality of photorealistic images created with StableDiffusion using the Yolo visualization system. For that it aims to identify a combination of parameters in Stable Diffusion that increments Yolo's visualization abilities.

## Stable Diffusion Parameters

The list of configuration parameters that StableYolo changes and their ranges are:
```  
"num_inference_steps" --> [0,100] Integer
"guidance_scale" --> range: [0, 20] Floating Point
"guidance_rescale"  --> range [0,1] Floating Point
"seed" --> range [0,512] Integer 
"prompt"  --> range: ['photograph', 'digital','color','Ultra Real', 'film grain', 'Kodak portra 800', 'Depth of field 100mm', 'overlapping compositions', 'blended visuals', 'trending on artstation', 'award winning']
"negative_prompt": --> range: ['illustration', 'painting', 'drawing', 'art', 'sketch','lowres', 'error', 'cropped', 'worst quality', 'low quality', 'jpeg artifacts', 'out of frame', 'watermark', 'signature']
```
## Dependencies for Yolo
```
pip3 install ultralytics
```

This should make Yolo work. You can test if it works by:
```
yolo version
```

Full information is here: https://github.com/ultralytics/ultralytics

## Depencencies for Stable Diffusion

This should install stable diffusion, make sure your graphics are working:
```
pip3 install diffusers transformers accelerate scipy safetensors
```

Just in case you might need pyTorch, although it should be in the previous Wheel:
```
pip3 install torch
```

## Some of the Extra Parameters

From the documentation of the stable diffusion API, we can see some parameters associated with the library. 

https://stablediffusionapi.com/docs/stable-diffusion-api/text2img/

Their description is in the webpage, but just in case:

- key:	Your API Key used for request authorization.
- prompt:	Text prompt with description of the things you want in the image to be generated.
- negative_prompt:	Items you don't want in the image.
- width	Max Height: Width: 1024x1024.
- height	Max Height: Width: 1024x1024.
- samples:	Number of images to be returned in response. The maximum value is 4.
- num_inference_steps:	Number of denoising steps. Available values: 21, 31, 41, 51.
- safety_checker:	A checker for NSFW images. If such an image is detected, it will be replaced by a blank image.
- enhance_prompt:	Enhance prompts for better results; default: yes, options: yes/no.
- seed:	Seed is used to reproduce results, same seed will give you same image in return again. Pass null for a random number.
- guidance_scale:	Scale for classifier-free guidance (minimum: 1; maximum: 20).
- multi_lingual:	Allow multi lingual prompt to generate images. Use "no" for the default English.
- panorama:	Set this parameter to "yes" to generate a panorama image.
- self_attention:	If you want a high quality image, set this parameter to "yes". In this case the image generation will take more time.
- upscale:	Set this parameter to "yes" if you want to upscale the given image resolution two times (2x). If the requested resolution is 512 x 512 px, the generated image will be 1024 x 1024 px.
- embeddings_model:	This is used to pass an embeddings model (embeddings_model_id).
- webhook:	Set an URL to get a POST API call once the image generation is complete.
- track_id:	This ID is returned in the response to the webhook API call. This will be used to identify the webhook request.

An example about how to set it up:
```
{
  "key": "",
  "prompt": "ultra realistic close up portrait ((beautiful pale cyberpunk female with heavy black eyeliner))",
  "negative_prompt": null,
  "width": "512",
  "height": "512",
  "samples": "1",
  "num_inference_steps": "20",
  "safety_checker": "no",
  "enhance_prompt": "yes",
  "seed": null,
  "guidance_scale": 7.5,
  "multi_lingual": "no",
  "panorama": "no",
  "self_attention": "no",
  "upscale": "no",
  "embeddings_model": null,
  "webhook": null,
  "track_id": null
}
```
