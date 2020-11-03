#include <stdio.h>
#include <math.h>
#include <stdlib.h>

void convnet_padhamsu_tensorflow(int* image, int* kernel, int stride, int* padmap, int image_length, int pad_length);

using namespace std;

int main()
{
	int image1[] = { 3, 2, 4, 5, 2, 3, 1, 0, 4, 2, 1, 4, 3, 4, 7, 2 };
	int kernel1[] = { 1, 1, 0, 0, 1, 0, 1, 0, 1 };
	int stride1[] = { 2,2 };

	int image_length = 4;
	int kernel_length = 3;
	int stride = stride1[0];									// 2
	int pad_length = stride * (image_length - 1) - image_length + kernel_length;	//5

	int* image = (int*)malloc(sizeof(image1));
	int* kernel = (int*)malloc(sizeof(kernel1));
	int* padmap = (int*)malloc((image_length + pad_length)*(image_length + pad_length) * sizeof(int));

	
	//printf("%d\n", pad_length*pad_length);
	//printf("%d\n", sizeof(image));
	//printf("%d\n", sizeof(kernel));
	//printf("%d\n", sizeof(padmap));


//
///*

	printf("%d\n", sizeof(image1));
	printf("%d\n", sizeof(kernel));



	for (int i = 0; i < sizeof(image1) / sizeof(int); i++)
	{
		image[i] = image1[i];
		//printf("%d\n", image[i]);

	}
	for (int i = 0; i < sizeof(kernel1) / sizeof(int); i++)
	{
		kernel[i] = kernel1[i];
		//printf("%d\n", kernel[i]);

	}
	for (int i = 0; i < (image_length + pad_length) * (image_length + pad_length); i++)
	{
		padmap[i] = 0;
		//printf("%d\n", padmap[i]);

	}

/*
	printf("%d\n", sizeof(image));
	printf("%d\n", sizeof(kernel));
	printf("%d\n", sizeof(padmap));
*/

	convnet_padhamsu_tensorflow(image, kernel, stride, padmap, image_length, pad_length);
	
	int cnt = 0;

	for (int i = 0; i < (image_length + pad_length) * (image_length + pad_length); i++)
	{	

		cnt += 1;

		printf("%d  ", padmap[i]);
		if ( cnt == (image_length + pad_length))
		{
			printf("\n");
			cnt = 0;
			
		}

	}

	free(image);
	free(kernel);
	free(padmap);


	return 0;


}


void convnet_padhamsu_tensorflow(int* image, int* kernel, int stride, int* padmap, int image_length, int pad_length)
{
	
	for (int pad_h = 0; pad_h < (image_length + pad_length); pad_h++)
	{
		for (int pad_w = 0; pad_w < (image_length + pad_length); pad_w++)
		{
			if (pad_h < pad_length / 2 or pad_h >= (image_length + pad_length / 2) or pad_w < pad_length / 2 or pad_w >= (image_length + pad_length / 2))
			{
				continue;
			}
			else
			{
				padmap[(image_length + pad_length) * pad_h + pad_w] = image[image_length * (pad_h - pad_length / 2) + (pad_w - pad_length / 2)];

			}
		}


	}
}