#include <iostream>
#include "opencv\cv.h"
#include "opencv\highgui.h"
#include <assert.h>
#include <time.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdio.h>
#include <cv.h>
#include <string>
#include <fstream>
#include <string>
#include <omp.h>
#include <math.h>
#include <opencv2/core/core.hpp>  
#include <opencv2/highgui/highgui.hpp>  

using namespace cv;
using namespace std;

// calculate entropy of an image
double Entropy(Mat img){
	double temp[256];
	for(int i=0;i<256;i++){
		temp[i] = 0.0;
	}
	for(int m=0;m<img.rows;m++){
		const uchar* t = img.ptr<uchar>(m);
		for(int n=0;n<img.cols;n++){
			int i = t[n];
			temp[i] = temp[i]+1;
		}
	}
	for(int i=0;i<256;i++){
		temp[i] = temp[i]/(img.rows*img.cols);
	}
	double result = 0;
	for(int i =0;i<256;i++){
		if(temp[i]==0.0)
			result = result;
		else
			result = result-temp[i]*(log(temp[i])/log(2.0));
	}
	return result;
}

void BGRtoY(Mat input, Mat Y){
	Y = Mat::zeros(input.rows,input.cols,CV_8UC1);
	double R=0.0, B=0.0, G=0.0;
	for(int j = 0;j < input.rows; j++){
		for(int i = 0;i < input.cols; i++){
			B=(double)(input.at<Vec3b>(j,i)[0]);	
			G=(double)(input.at<Vec3b>(j,i)[1]);	
			R=(double)(input.at<Vec3b>(j,i)[2]);

			Y.ptr<uchar>(j)[i] = (uchar)(0.299*R+0.587*G+0.114*B);
		}
	}
}

void fastmean(Mat input,double men[3],float ga){
	double R=0.0, B=0.0, G=0.0, temp=0.0;
	double pop = input.rows*input.cols;
	float va = 1/ga;
	int vara = va;

	for(int i = 0;i < pop; i=i+va){
		temp++;
		B+=(double)(input.data[i*3]);	
		G+=(double)(input.data[i*3+1]);	
		R+=(double)(input.data[i*3+2]);
	}
	men[0] = R/temp;
	men[1] = G/temp;
	men[2] = B/temp;
}


void pwr_enhancerV2(Mat input, Mat output, double mu[3], double men[3]){

	double deltaR = 0.0, deltaG = 0.0, deltaB = 0.0;
	deltaR = mu[0] * men[0];
	deltaG = mu[1] * men[1];
	deltaB = mu[2] * men[2];

	double R, G, B;
	for (int j = 0; j < input.rows; j++){
		for (int i = 0; i < input.cols; i++){
			
			B = (double)input.at<Vec3b>(j,i)[0] - deltaB; 
			if (B < 0.0){B = 0.0;}
			else if (B > 255.0){B = 255.0;} else{}
			output.at<Vec3b>(j,i)[0] = (uchar)(cvRound(B));
			G = (double)input.at<Vec3b>(j,i)[1] - deltaG; 
			if (G < 0.0){G = 0.0;}
			else if (G > 255.0){G = 255.0;} else{}
			output.at<Vec3b>(j,i)[1] = (uchar)(cvRound(G));
			R = (double)input.at<Vec3b>(j,i)[2] - deltaR; 
			if (R < 0.0){R = 0.0;}
			else if (R > 255.0){R = 255.0;} else{}
			output.at<Vec3b>(j,i)[2] = (uchar)(cvRound(R));		
		}}
}

void pwr_enhancerfV2(Mat input, Mat output, double mu[3], double men[3]){
	#pragma omp parallel
	{
	#pragma omp sections nowait
	{
		#pragma omp section
		{
			double deltaR = 0.0;
			deltaR = mu[0] * men[0];
			double R, G, B;
			for (int j = 0; j < input.rows; j++){
				for (int i = 0; i < input.cols; i++){
					R = (double)input.at<Vec3b>(j,i)[2] - deltaR; 
					if (R < 0.0){R = 0.0;}
					else if (R > 255.0){R = 255.0;} 
					else{}
					output.at<Vec3b>(j,i)[2] = (uchar)(cvRound(R));		
				}
			}
		}
		#pragma omp section
		{
			double deltaG = 0.0;
			deltaG = mu[1] * men[1];
			double R, G, B;
			for (int j = 0; j < input.rows; j++){
				for (int i = 0; i < input.cols; i++){
					G = (double)input.at<Vec3b>(j,i)[1] - deltaG; 
					if (G < 0.0){G = 0.0;}
					else if (G > 255.0){G = 255.0;} 
					else{}
					output.at<Vec3b>(j,i)[1] = (uchar)(cvRound(G));
				}
			}
		}
		#pragma omp section
		{
			double deltaB = 0.0;
			deltaB = mu[2] * men[2];
			double R, G, B;
			for (int j = 0; j < input.rows; j++){
				for (int i = 0; i < input.cols; i++){
					B = (double)input.at<Vec3b>(j,i)[0] - deltaB; 
					if (B < 0.0){B = 0.0;}
					else if (B > 255.0){B = 255.0;}
					else{}
					output.at<Vec3b>(j,i)[0] = (uchar)(cvRound(B));
				}
			}
		}
	}}
}

void set_parameter(Mat frame, double para[4]){
	Mat gray = Mat::zeros(frame.rows,frame.cols, CV_8UC1);
	BGRtoY(frame, gray);
	
	double Air = 0.0;
	double mean = 0.0;
	double OR_th = 0.0;
	double sd = 0.0;
	double pop = (double)(frame.rows*frame.cols);
	for (int j = 0; j < frame.rows; j++){
		for (int i = 0; i < frame.cols; i++){
			mean += (double)gray.ptr<uchar>(j)[i];
		}
	}
	mean /= pop;
	for (int j = 0; j < frame.rows; j++){
		for (int i = 0; i < frame.cols; i++){
			OR_th += pow(((double)gray.ptr<uchar>(j)[i]-mean),2);
		}
	}
	OR_th /= pop;
	sd = sqrt(OR_th);
	OR_th = mean + (0.5*sqrt(OR_th));

	if (OR_th>255){
		OR_th = 255;
	}

	double beta = 25.0;
	
	if (OR_th>250){
		Air = OR_th;
	}
	else{
		Air = (255.0 + (OR_th-beta))/2;
	}

	para[0] = mean;
	para[1] = OR_th;
	para[2] = Air;
	para[3] = sd;
}

int main()
{
	// read video
	VideoCapture vc("C:/Users/NTUST-LiMing/Desktop/2016matlabcode/algorithm/t_video/Avatar - Eywa.mp4");
	//VideoCapture vc("C:/Users/NTUST-LiMing/Downloads/NBA.mp4");
	int fps = vc.get(CV_CAP_PROP_FPS);
	int fH = vc.get(CV_CAP_PROP_FRAME_HEIGHT);
	int fW = vc.get(CV_CAP_PROP_FRAME_WIDTH);

	//create output video
	VideoWriter vm("Power_saving_mine_Avatart.avi", CV_FOURCC('M', 'J', 'P', 'G'), fps, Size(fW, fH));
	if (vc.isOpened() == false) {
		cout << "Fail to open video" << endl;
		system("pause");
		return 1;
	}
	if (vm.isOpened() == false) {
		cout << "Fail to write video" << endl;
		system("pause");
		return 1;
	}

	char filename[]="Entropy.txt";
	fstream fp;
	fp.open(filename, ios::out);

	//create window
	namedWindow("show", CV_WINDOW_AUTOSIZE);
	namedWindow("output2", CV_WINDOW_AUTOSIZE);

	//RGB to Y
	Mat frame, dist, dist2;
	double mean  = 0.0, OR_th  = 0.0, Air  = 0.0, sd = 0.0, cov_rate = 0.0;
	vc >> frame;

	double E = 0.0;

	double para[4];//[0] mean [1] OR_th [2] Air
	set_parameter(frame, para);
	mean = para[0];
	OR_th = para[1];
	Air = para[2] + 1;
	sd = para[3];
	
	if (mean>128){
		cov_rate = sd/128;
	}
	else{
		cov_rate = sd/mean;
	}

	// RGB mean
	double men[3];
	// RGB parameter
	double mu[3];
	// RGB fast mean
	double menf[3];
	float ga = 0.05;
	
	char key;
	char pause;
	int count = 0;
	double tempE = Entropy(frame);
	double dE = 0.0;
	double rateE = 0.0;
	while (frame.empty() == false) 
	{
		//count frame number
		count = count + 1;
		//show oringinal image
		imshow("show", frame);

		//Acquires input
		Mat input = frame;
		Mat output2 = Mat::zeros(input.rows,input.cols,input.type());

		int start, stop;
		start = clock();
		
		E = Entropy(frame);
		dE = E - tempE;
		if (dE<0){
			dE = -dE;
		}
		rateE = dE/(tempE+0.0001);
		if (rateE>0.05){
			tempE = E;
			set_parameter(frame, para);
			mean = para[0];
			OR_th = para[1];
			Air = para[2] + 1;
			sd = para[3];
		}

		////開啟檔案
		//if(!fp){
		//	//如果開啟檔案失敗，fp為0；成功，fp為非0
		//	cout<<"Fail to open file: "<<filename<<endl;
		//}
		//else{
		//	fp<< E <<endl;
		//}
		
		// fast mean
		fastmean(frame ,men ,ga);

		double tR = 0.0, tG = 0.0, tB = 0.0;
		tR = 1-(OR_th/255)*(men[0]/Air);
		tG = 1-(OR_th/255)*(men[1]/Air);
		tB = 1-(OR_th/255)*(men[2]/Air);

		double Rmu = 0.0, Gmu = 0.0, Bmu = 0.0;

		Rmu = ((Air - men[0])*(1-tR))/(tR*mean);
		Gmu = ((Air - men[1])*(1-tG))/(tG*mean);
		Bmu = ((Air - men[2])*(1-tB))/(tB*mean);

		if (Rmu>0.5)
			Rmu = Rmu*cov_rate;
		if (Gmu>0.5)
			Gmu = Gmu*cov_rate;
		if (Bmu>0.5)
			Bmu = Bmu*cov_rate;

		mu[0] = Rmu;
		mu[1] = Gmu;
		mu[2] = Bmu;

		//cout << men[2] << " " << tB << " " << Bmu << " " << mu[2] << endl;

		pwr_enhancerfV2(input, output2, mu, men);

		stop = clock();
		cout<<"Time="<<(stop-start)/(double)(CLOCKS_PER_SEC)<<" seconds"<<endl;
	
		key = waitKey(33);
		/*if(key==27)
			pause=!pause;
        while(pause)
        {
            key=waitKey(1000);
            if(key=='t')
            pause=!pause;
        }*/
		if (key == 'b')
			break;
		int maxframe = fps*60;
		if (count==maxframe)
			break;

		resize(output2, dist2, Size(fW, fH));
		imshow("output2", dist2);
		vm << dist2;

		vc >> frame;
		
	}
	return 0;
}