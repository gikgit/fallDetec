#include <iostream>
#include <stdio.h>
#include <math.h>

#include "cxcore.h"
#include "highgui.h"
#include "cv.h"
#include "cvaux.h"
#include "ml.h"


#include "const.h"
#include "func.h"

using namespace cv;
using namespace std;

/*-----------------------------------------*/
/*            Main function                */
/*-----------------------------------------*/

int main(int argc, char *argv[])
{ 
    
    /*-------------------------Cam1 Parameters----------------------------*/
    point cam1_mean0, cam1_mean1;
    cam1_mean0.ratio = 0, cam1_mean0.vitess = 0;
    cam1_mean1.ratio = 0, cam1_mean1.vitess = 0;
    Mat cam1_var0 = Mat::zeros(2,2,CV_32F);
    Mat cam1_var1 = Mat::zeros(2,2,CV_32F);
    float cam1_p0 = 0, cam1_p1 = 0;

    TrainingCam1( cam1_mean0, cam1_mean1, cam1_var0, cam1_var1, cam1_p0, cam1_p1 );

    
    printf("---------------cam %d--------------------\n",1);
    printf("p0 = %f\tp1 = %f\n",cam1_p0, cam1_p1);
    printf("mean0 = %f\t%f\n",cam1_mean0.ratio, cam1_mean0.vitess);
    printf("mean1 = %f\t%f\n",cam1_mean1.ratio, cam1_mean1.vitess);
    printf("var0 = %f\t%f\t%f\t%f\n",cam1_var0.at<float>(0,0), cam1_var0.at<float>(0,1), cam1_var0.at<float>(1,0), cam1_var0.at<float>(1,1));
    printf("var1 = %f\t%f\t%f\t%f\n",cam1_var1.at<float>(0,0), cam1_var1.at<float>(0,1), cam1_var1.at<float>(1,0), cam1_var1.at<float>(1,1));

    
    /*-------------------------Cam2 Parameters----------------------------*/  
    point cam2_mean0, cam2_mean1;
    cam2_mean0.ratio = 0, cam2_mean0.vitess = 0;
    cam2_mean1.ratio = 0, cam2_mean1.vitess = 0;
    Mat cam2_var0 = Mat::zeros(2,2,CV_32F);
    Mat cam2_var1 = Mat::zeros(2,2,CV_32F);
    float cam2_p0 = 0, cam2_p1 = 0;
  
    
    TrainingCam2( cam2_mean0, cam2_mean1, cam2_var0, cam2_var1, cam2_p0, cam2_p1 );
 
    printf("---------------cam %d--------------------\n",2);
    printf("p0 = %f\tp1 = %f\n",cam2_p0, cam2_p1);
    printf("mean0 = %f\t%f\n",cam2_mean0.ratio, cam2_mean0.vitess);
    printf("mean1 = %f\t%f\n",cam2_mean1.ratio, cam2_mean1.vitess);
    printf("var0 = %f\t%f\t%f\t%f\n",cam2_var0.at<float>(0,0), cam2_var0.at<float>(0,1), cam2_var0.at<float>(1,0), cam2_var0.at<float>(1,1));
    printf("var1 = %f\t%f\t%f\t%f\n",cam2_var1.at<float>(0,0), cam2_var1.at<float>(0,1), cam2_var1.at<float>(1,0), cam2_var1.at<float>(1,1));

    
    /*-------------------------Cam3 Parameters-----------------------------*/
    point cam3_mean0, cam3_mean1;
    Mat cam3_var0 = Mat::zeros(2,2,CV_32F);
    Mat cam3_var1 = Mat::zeros(2,2,CV_32F);
    float cam3_p0, cam3_p1;

    TrainingCam3( cam3_mean0, cam3_mean1, cam3_var0, cam3_var1, cam3_p0, cam3_p1 );

    printf("---------------cam %d--------------------\n",3);
    printf("p0 = %f\tp1 = %f\n",cam3_p0, cam3_p1);
    printf("mean0 = %f\t%f\n",cam3_mean0.ratio, cam3_mean0.vitess);
    printf("mean1 = %f\t%f\n",cam3_mean1.ratio, cam3_mean1.vitess);
    printf("var0 = %f\t%f\t%f\t%f\n",cam3_var0.at<float>(0,0), cam3_var0.at<float>(0,1), cam3_var0.at<float>(1,0), cam3_var0.at<float>(1,1));
    printf("var1 = %f\t%f\t%f\t%f\n",cam3_var1.at<float>(0,0), cam3_var1.at<float>(0,1), cam3_var1.at<float>(1,0), cam3_var1.at<float>(1,1));  

   
    /*-------------------------Cam4 Parameters-----------------------------*/
    point cam4_mean0, cam4_mean1;
    Mat cam4_var0 = Mat::zeros(2,2,CV_32F);
    Mat cam4_var1 = Mat::zeros(2,2,CV_32F);
    float cam4_p0, cam4_p1;

    TrainingCam4( cam4_mean0, cam4_mean1, cam4_var0, cam4_var1, cam4_p0, cam4_p1 );

    printf("---------------cam %d--------------------\n",4);
    printf("p0 = %f\tp1 = %f\n",cam4_p0, cam4_p1);
    printf("mean0 = %f\t%f\n",cam4_mean0.ratio, cam4_mean0.vitess);
    printf("mean1 = %f\t%f\n",cam4_mean1.ratio, cam4_mean1.vitess);
    printf("var0 = %f\t%f\t%f\t%f\n",cam4_var0.at<float>(0,0), cam4_var0.at<float>(0,1), cam4_var0.at<float>(1,0), cam4_var0.at<float>(1,1));
    printf("var1 = %f\t%f\t%f\t%f\n",cam4_var1.at<float>(0,0), cam4_var1.at<float>(0,1), cam4_var1.at<float>(1,0), cam4_var1.at<float>(1,1));  




/*--------------------------------------------------Choose test scene------------------------------------------------------------------------------------*/

    /*------------------------Test 11------------------------------*/
    // int test = 11;
    
    // int cam1_start_frames_test[12] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 89};
    // int cam1_start_bg_test[12] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 20};                              
    // int cam1_end_bg_test[12] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 78};
                                  
    // int cam1_sum_nb_test = 0;
    // int cam1_end_frames_test[12] = {};
    // int cam1_total_frames_test[12] = {};
   
    // int cam2_start_frames_test[12] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 91};
    // int cam2_start_bg_test[12] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 16};                              
    // int cam2_end_bg_test[12] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 72};
                                  
    // int cam2_sum_nb_test = 0;
    // int cam2_end_frames_test[12] = {};
    // int cam2_total_frames_test[12] = {};

    // int cam3_start_frames_test[12] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 89};
    // int cam3_start_bg_test[12] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 20};                              
    // int cam3_end_bg_test[12] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 72};
                                      
    // int cam3_sum_nb_test = 0;
    // int cam3_end_frames_test[12] = {};
    // int cam3_total_frames_test[12] = {};

    // int cam4_start_frames_test[12] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 88};
    // int cam4_start_bg_test[12] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 27};                              
    // int cam4_end_bg_test[12] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 56};

    // int cam4_sum_nb_test = 0;
    // int cam4_end_frames_test[12] = {};
    // int cam4_total_frames_test[12] = {};

    //------------------------Test 12-----------------------------
    // int test = 12;
    
    // int cam1_start_frames_test[13] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 84};
    // int cam1_start_bg_test[13] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 10};                              
    // int cam1_end_bg_test[13] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 70};
                                  
    // int cam1_sum_nb_test = 0;
    // int cam1_end_frames_test[13] = {};
    // int cam1_total_frames_test[13] = {};

    // int cam2_start_frames_test[13] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 85};
    // int cam2_start_bg_test[13] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 20};                              
    // int cam2_end_bg_test[13] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 66};
                                  
    // int cam2_sum_nb_test = 0;
    // int cam2_end_frames_test[13] = {};
    // int cam2_total_frames_test[13] = {};

    // int cam3_start_frames_test[13] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 85};
    // int cam3_start_bg_test[13] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 17};                              
    // int cam3_end_bg_test[13] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 69};
                                  
    // int cam3_sum_nb_test = 0;
    // int cam3_end_frames_test[13] = {};
    // int cam3_total_frames_test[13] = {};

    // int cam4_start_frames_test[13] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 83};
    // int cam4_start_bg_test[13] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 26};                              
    // int cam4_end_bg_test[13] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 53};
                                  
    // int cam4_sum_nb_test = 0;
    // int cam4_end_frames_test[13] = {};
    // int cam4_total_frames_test[13] = {};
  
    /*------------------------Test 13-----------------------------*/
    int test = 13;    
 
    int cam1_start_frames_test[14] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 75};
    int cam1_start_bg_test[14] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 10};                              
    int cam1_end_bg_test[14] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 66};
                                  
    int cam1_sum_nb_test = 0;
    int cam1_end_frames_test[14] = {};
    int cam1_total_frames_test[14] = {};

    int cam2_start_frames_test[14] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 70};
    int cam2_start_bg_test[14] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 10};                              
    int cam2_end_bg_test[14] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 52};
                                  
    int cam2_sum_nb_test = 0;
    int cam2_end_frames_test[14] = {};
    int cam2_total_frames_test[14] = {};

    int cam3_start_frames_test[14] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 75};
    int cam3_start_bg_test[14] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 13};                              
    int cam3_end_bg_test[14] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 59};
                                  
    int cam3_sum_nb_test = 0;
    int cam3_end_frames_test[14] = {};
    int cam3_total_frames_test[14] = {};

    int cam4_start_frames_test[14] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 73};
    int cam4_start_bg_test[14] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 18};                              
    int cam4_end_bg_test[14] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 48};
                                  
    int cam4_sum_nb_test = 0;
    int cam4_end_frames_test[14] = {};
    int cam4_total_frames_test[14] = {};
      /*--------------------------------------------------------------*/
    printf("-----------------testing scene %d-----------------\n", test);


    statis(test, test, cam1_start_frames_test, cam1_end_frames_test, cam1_total_frames_test, 1, cam1_sum_nb_test);
    statis(test, test, cam2_start_frames_test, cam2_end_frames_test, cam2_total_frames_test, 2, cam2_sum_nb_test);
    statis(test, test, cam3_start_frames_test, cam3_end_frames_test, cam3_total_frames_test, 3, cam3_sum_nb_test);
    statis(test, test, cam4_start_frames_test, cam4_end_frames_test, cam4_total_frames_test, 4, cam4_sum_nb_test);

    
    float cam1_label_test[cam1_sum_nb_test];
    float cam2_label_test[cam2_sum_nb_test];
    float cam3_label_test[cam3_sum_nb_test];
    float cam4_label_test[cam4_sum_nb_test];
    
    point* cam1_frames_test = (point *) malloc (cam1_sum_nb_test*sizeof(point));
    point* cam2_frames_test = (point *) malloc (cam2_sum_nb_test*sizeof(point));
    point* cam3_frames_test = (point *) malloc (cam3_sum_nb_test*sizeof(point));
    point* cam4_frames_test = (point *) malloc (cam4_sum_nb_test*sizeof(point));

    loadData(test, test, cam1_start_bg_test, cam1_end_bg_test, cam1_start_frames_test, 
                cam1_end_frames_test, cam1_total_frames_test, cam1_label_test, 1, cam1_sum_nb_test, cam1_frames_test);

    loadData(test, test, cam2_start_bg_test, cam2_end_bg_test, cam2_start_frames_test, 
                cam2_end_frames_test, cam2_total_frames_test, cam2_label_test, 2, cam2_sum_nb_test, cam2_frames_test);
    
    loadData(test, test, cam3_start_bg_test, cam3_end_bg_test, cam3_start_frames_test, 
                cam3_end_frames_test, cam3_total_frames_test, cam3_label_test, 3, cam3_sum_nb_test, cam3_frames_test);

    loadData(test, test, cam4_start_bg_test, cam4_end_bg_test, cam4_start_frames_test, 
                cam4_end_frames_test, cam4_total_frames_test, cam4_label_test, 4, cam4_sum_nb_test, cam4_frames_test);


    float mean0_cam1[] = { cam1_mean0.ratio, cam1_mean0.vitess };
    float mean1_cam1[] = { cam1_mean1.ratio, cam1_mean1.vitess };
    float var0_cam1[] = { cam1_var0.at<float>(0,0), cam1_var0.at<float>(0,1), cam1_var0.at<float>(1,0), cam1_var0.at<float>(1,1)};
    float var1_cam1[] = { cam1_var1.at<float>(0,0), cam1_var1.at<float>(0,1), cam1_var1.at<float>(1,0), cam1_var1.at<float>(1,1)};

    CvMat mean_0_cam1 = cvMat(2, 1, CV_32F, mean0_cam1);
    CvMat mean_1_cam1 = cvMat(2, 1, CV_32F, mean1_cam1);
    CvMat cov_0_cam1 = cvMat(2, 2, CV_32F, var0_cam1);
    CvMat cov_1_cam1 = cvMat(2, 2, CV_32F, var1_cam1);

    float mean0_cam2[] = { cam2_mean0.ratio, cam2_mean0.vitess };
    float mean1_cam2[] = { cam2_mean1.ratio, cam2_mean1.vitess };
    float var0_cam2[] = { cam2_var0.at<float>(0,0), cam2_var0.at<float>(0,1), cam2_var0.at<float>(1,0), cam2_var0.at<float>(1,1)};
    float var1_cam2[] = { cam2_var1.at<float>(0,0), cam2_var1.at<float>(0,1), cam2_var1.at<float>(1,0), cam2_var1.at<float>(1,1)};

    CvMat mean_0_cam2 = cvMat(2, 1, CV_32F, mean0_cam2);
    CvMat mean_1_cam2 = cvMat(2, 1, CV_32F, mean1_cam2);
    CvMat cov_0_cam2 = cvMat(2, 2, CV_32F, var0_cam2);
    CvMat cov_1_cam2 = cvMat(2, 2, CV_32F, var1_cam2);

    float mean0_cam3[] = { cam3_mean0.ratio, cam3_mean0.vitess };
    float mean1_cam3[] = { cam3_mean1.ratio, cam3_mean1.vitess };
    float var0_cam3[] = { cam3_var0.at<float>(0,0), cam3_var0.at<float>(0,1), cam3_var0.at<float>(1,0), cam3_var0.at<float>(1,1)};
    float var1_cam3[] = { cam3_var1.at<float>(0,0), cam3_var1.at<float>(0,1), cam3_var1.at<float>(1,0), cam3_var1.at<float>(1,1)};

    CvMat mean_0_cam3 = cvMat(2, 1, CV_32F, mean0_cam3);
    CvMat mean_1_cam3 = cvMat(2, 1, CV_32F, mean1_cam3);
    CvMat cov_0_cam3 = cvMat(2, 2, CV_32F, var0_cam3);
    CvMat cov_1_cam3 = cvMat(2, 2, CV_32F, var1_cam3);

    float mean0_cam4[] = { cam4_mean0.ratio, cam4_mean0.vitess };
    float mean1_cam4[] = { cam4_mean1.ratio, cam4_mean1.vitess };
    float var0_cam4[] = { cam4_var0.at<float>(0,0), cam4_var0.at<float>(0,1), cam4_var0.at<float>(1,0), cam4_var0.at<float>(1,1)};
    float var1_cam4[] = { cam4_var1.at<float>(0,0), cam4_var1.at<float>(0,1), cam4_var1.at<float>(1,0), cam4_var1.at<float>(1,1)};

    CvMat mean_0_cam4 = cvMat(2, 1, CV_32F, mean0_cam4);
    CvMat mean_1_cam4 = cvMat(2, 1, CV_32F, mean1_cam4);
    CvMat cov_0_cam4 = cvMat(2, 2, CV_32F, var0_cam4);
    CvMat cov_1_cam4 = cvMat(2, 2, CV_32F, var1_cam4);
   
   /*---------------------------------------------------------------------------------------------------------*/ 
    int min[] = {cam1_sum_nb_test, cam2_sum_nb_test, cam3_sum_nb_test, cam4_sum_nb_test};

    int temp = cam1_sum_nb_test;
    for(int i=1; i<4; i++){
       if (min[i]<temp){
           temp = min[i];
       } 
    }    

    classifier* vote =  (classifier *) malloc (temp*sizeof(classifier));
    
    for(int i=0; i<temp; i++){
      
       float cam1_prob0, cam1_prob1;
       float cam1_v[] = {cam1_frames_test[i].ratio, cam1_frames_test[i].vitess };
       CvMat cam1_vec_test = cvMat(2, 1, CV_32F, cam1_v);

       float cam2_prob0, cam2_prob1;
       float cam2_v[] = {cam2_frames_test[i].ratio, cam2_frames_test[i].vitess };
       CvMat cam2_vec_test = cvMat(2, 1, CV_32F, cam2_v);

       float cam3_prob0, cam3_prob1;
       float cam3_v[] = {cam3_frames_test[i].ratio, cam3_frames_test[i].vitess };
       CvMat cam3_vec_test = cvMat(2, 1, CV_32F, cam3_v);

       float cam4_prob0, cam4_prob1;
       float cam4_v[] = {cam4_frames_test[i].ratio, cam4_frames_test[i].vitess };
       CvMat cam4_vec_test = cvMat(2, 1, CV_32F, cam4_v);
       
       cam1_prob0 = cvGaussPdf( &cam1_vec_test, &mean_0_cam1, &cov_0_cam1, true, false);
       cam1_prob1 = cvGaussPdf( &cam1_vec_test, &mean_1_cam1, &cov_1_cam1, true, false);

       cam2_prob0 = cvGaussPdf( &cam2_vec_test, &mean_0_cam2, &cov_0_cam2, true, false);
       cam2_prob1 = cvGaussPdf( &cam2_vec_test, &mean_1_cam2, &cov_1_cam2, true, false);

       cam3_prob0 = cvGaussPdf( &cam3_vec_test, &mean_0_cam3, &cov_0_cam3, true, false);
       cam3_prob1 = cvGaussPdf( &cam3_vec_test, &mean_1_cam3, &cov_1_cam3, true, false);

       cam4_prob0 = cvGaussPdf( &cam4_vec_test, &mean_0_cam4, &cov_0_cam4, true, false);
       cam4_prob1 = cvGaussPdf( &cam4_vec_test, &mean_1_cam4, &cov_1_cam4, true, false);


       if(cam1_p0*cam1_prob0 > cam1_p1*cam1_prob1 ){
          vote[i].x1 = 0;
       }else{
          vote[i].x1 = 1;
       }
      
       if(cam2_p0*cam2_prob0 > cam2_p1*cam2_prob1 ){
          vote[i].x2 = 0;
       }else{
          vote[i].x2 = 1;
       }

       if(cam3_p0*cam3_prob0 > cam3_p1*cam3_prob1 ){
          vote[i].x3 = 0;
       }else{
          vote[i].x3 = 1;
       }
   
          
       if(cam4_p0*cam4_prob0 > cam4_p1*cam4_prob1 ){
       		vote[i].x4 = 0;
       }else{
          vote[i].x4 = 1;
       }





       // if(cam1_p0*cam1_prob0 > cam1_p1*cam1_prob1 && fabs(cam1_frames_test[i].vitess)<=0.07){
       //    vote[i].x1 = 0;
       // }else{
       //    vote[i].x1 = 1;
       // }
      
       // if(cam2_p0*cam2_prob0 > cam2_p1*cam2_prob1 && fabs(cam2_frames_test[i].vitess)<=0.07){
       //    vote[i].x2 = 0;
       // }else{
       //    vote[i].x2 = 1;
       // }

       // if(cam3_p0*cam3_prob0 > cam3_p1*cam3_prob1 && fabs(cam3_frames_test[i].vitess)<=0.07){
       //    vote[i].x3 = 0;
       // }else{
       //    vote[i].x3 = 1;
       // }
   
          
       // if(cam4_p0*cam4_prob0 > cam4_p1*cam4_prob1 && fabs(cam4_frames_test[i].vitess)<=0.07){
       // 		vote[i].x4 = 0;
       // }else{
       //    vote[i].x4 = 1;
       // }

      
       if(vote[i].x1 + vote[i].x2 + vote[i].x3 + vote[i].x4 >= 3   ) printf("%d\tnot Fall\n",i);
       else printf("%d\tFall\n",i);
    }
   
   
  /*---------------------------------------------------------------------------*/        
    return 0;
}









/*------------------------------------*/
/*    Calculate background            */
/*------------------------------------*/
IplImage* bg(int index, int nb_cam, int start, int end)
{
      
  int selectedFrames[end-start];
  char fname[260];
  int key = 0;
  const CvSize sz= cvSize(320,240);
        
  IplImage* bg = cvCreateImage(sz,IPL_DEPTH_8U,1);
  IplImage* cumul = cvCreateImage(sz,IPL_DEPTH_64F,1);
  IplImage* tmp = cvCreateImage(sz,IPL_DEPTH_64F,1);

  for(int i=0;i<end-start;i++) selectedFrames[i]=start+i;

  cvZero(cumul);
  int nb=0;
   
  for(int i=0; i<end-start; i++){
    sprintf(fname,"%02d/cam%01d/%03d.jpg",index,nb_cam,selectedFrames[i]);
    IplImage*  img = cvLoadImage(fname,0); 
    if(img){	
      cvCvtScale( img, tmp, 1, 0 );
      cvAcc(tmp,cumul);
      nb++; 
      cvConvertScaleAbs(cumul,bg, 1.0/nb);  
      cvReleaseImage(&img);	
    }          
  }
        
  cvReleaseImage(&cumul);
  cvReleaseImage(&tmp);

  return bg;
}

/*-----------------------------------------------------------------------*/
/*      Component Connection Algorithem  provided by Mohamed Dahmane     */
/*-----------------------------------------------------------------------*/
int etiquetage(IplImage* imageBin, CvMat* &matEtiq)
{

  int i, j;
  int nl, nc, numEtiq, minEtiq, maxEtiq, nbMaxEtiq;
  int* T=NULL;
  int nbEtiq;

  nl = imageBin->height;
  nc = imageBin->width;
  
  matEtiq = cvCreateMat(nl, nc, CV_32S);
  
  numEtiq = 0;

  nbMaxEtiq = nl*nc;
  T = new int[nbMaxEtiq];
  for(i = 0; i < nbMaxEtiq; i++) T[i] = i;
  

  cvZero(matEtiq);
  int step = matEtiq->step/sizeof(int);

  for(i = 1; i < nl; i++){
    char* currentLine = imageBin->imageData + i*imageBin->widthStep;
    for(j = 1; j < nc; j++){
      unsigned char* currentPixel = (unsigned char*) currentLine+j;
      if(*currentPixel != 0){
        int etiquetteN = *(matEtiq->data.i + (i-1)*step + j);
	int etiquetteW = *(matEtiq->data.i + (i*step + j-1));
        unsigned char pixN = *(currentPixel-imageBin->widthStep);
        unsigned char pixW = *(currentPixel-1);
        if(i==1) pixN = 0;
        if(j==1) pixW = 0;
	  if( pixN==0 && pixW==0) *(matEtiq->data.i + i*step + j) = ++numEtiq;
	  else {	
	     if( etiquetteN == etiquetteW && etiquetteN!=0 ) *(matEtiq->data.i + i*step + j) = etiquetteW;
	     else {				
	       if( etiquetteW<etiquetteN ) {
	         if( etiquetteW!=0 ) {
		    minEtiq = etiquetteW; maxEtiq = etiquetteN;
	         }
	         else {
		    minEtiq = etiquetteN; maxEtiq = etiquetteW;
	         }
	       }
	       else {
		 if( etiquetteN!=0 ) {
		    minEtiq = etiquetteN; maxEtiq = etiquetteW;
		 }
		 else {
		    minEtiq = etiquetteW; maxEtiq = etiquetteN;
		 }
	       }

	       *(matEtiq->data.i + i*step + j) = T[minEtiq];	       
	       if (maxEtiq!=0)
	       if(T[maxEtiq] != T[minEtiq]){
	         if(T[maxEtiq]==maxEtiq) T[maxEtiq] = T[minEtiq];
		 else do{
                    int k = T[maxEtiq];
                    T[maxEtiq] = T[minEtiq];
                    maxEtiq = k;                
                 } while (T[maxEtiq]!=maxEtiq );
	       }
	     }
           }
        }
      }
   }

   for( i = 1; i <= numEtiq; i++) {
     j = i;
     while(T[j] != j){
       j = T[j];
       T[i]=j;
     }
   } 

   int m, n=1;
   i=1;
   while(i<=numEtiq){
     if(T[i]<=n) i++;
     else {
       m = T[i];
       n++;
       for( j = i; j <= numEtiq; j++) {
         if(T[j]==m)
	 T[j]=n;
	 i++;
       }
     }
   }

   nbEtiq = n;
   for(i = 1; i < nl; i++){
     for(j = 1; j < nc; j++){
      if( (*(matEtiq->data.i + i*nc + j)) != 0) 
        *(matEtiq->data.i + i*step + j) = T[(int)(*(matEtiq->data.i + i*step + j))];
     }
   }

   delete[] T;
   return nbEtiq;
}


void convertEtiq(CvMat* matEtiq, IplImage* &imgEtiq, int nbEtiq)
{ 
  int i, j, nl, nc;

  imgEtiq = cvCreateImage(cvGetSize(matEtiq), IPL_DEPTH_8U, 3);
  
  nl = matEtiq->height;
  nc = matEtiq->width;
  
  int step = (matEtiq->step/sizeof(int));

  double  fact =  255.*255.*255./nbEtiq;

  cvZero(imgEtiq);

  for(i = 1; i < nl; i++){
      char* currentLine = imgEtiq->imageData + i*imgEtiq->widthStep;
      for(j = 1; j < nc; j++){
          char* currentPixel = currentLine+imgEtiq->nChannels*j; 
          long Value = (long)  (*(matEtiq->data.i + i*step + j))* fact;
	  currentPixel[0]= (unsigned char) (Value % 256);		
          currentPixel[1]= (unsigned char) ((Value / 256) % 256);		  
	  currentPixel[2]= (unsigned char) ((Value / 256 / 256) % 256);
      }
   }
}

/*-----------------------------------------------*/
/*    Select the largest segmentation            */
/*-----------------------------------------------*/

void select(IplImage* imgEtiq, float &xmax, float &xmin, float &ymax, float &ymin )
{

 
  int i,j;

  float statistics[256];
  int val,val1;

  uchar *pImg   = ( uchar* )imgEtiq->imageData;

  for (int i = 0; i < 256; ++i) statistics[i] = 0;
 
  for(i = 0; i < imgEtiq->height; i++){
    for(j = 0; j < imgEtiq->width; j++){
      val = pImg[i*imgEtiq->widthStep + j*imgEtiq->nChannels];
      if (val!=0) statistics[val]++;
    }
  }

  
  float max = statistics[1];
  int max_i = 1;
  for(i=1; i <= 255; i++){
    if(statistics[i] > max){
      max = statistics[i];
      max_i = i;
    }
  }

  float* xlist;
  float* ylist;
  xlist = new float[(int)max];
  ylist = new float[(int)max];
  for (int i = 0; i < (int)max; ++i){
  	xlist[i] = 0.0;
  	ylist[i] = 0.0;
  }

  int index = 0;
  for(i = 0; i < imgEtiq->height-1; i++){
    for(j = 0; j < imgEtiq->width-1; j++){
      val1 = pImg[i*imgEtiq->widthStep + j*imgEtiq->nChannels];
      if (val1 == max_i){
        xlist[index] = j;
        ylist[index] = i;
        index++;
      }
    }
  }
  
  xmax = xlist[0];
  xmin = xlist[0];
  ymax = ylist[0];
  ymin = ylist[0];
  for(i = 1; i < max; i++){
    if(xmax < xlist[i]) xmax = xlist[i];
    if(xmin > xlist[i]) xmin = xlist[i];
    if(ymax < ylist[i]) ymax = ylist[i];
    if(ymin > ylist[i]) ymin = ylist[i];
  }

  cvRectangle(imgEtiq,                   
                cvPoint((int)xmin, (int)ymin),      
                cvPoint((int)xmax, (int)ymax),     
                cvScalar(255, 0, 0, 0), 
                1, 8, 0); 

  delete[] xlist;
  delete[] ylist;
  
}

/*----------------------------------------------*/
/*   Calculate  the ratio: width/height         */
/*----------------------------------------------*/
float ratio1(float xmax, float xmin, float ymax, float ymin)
{
	return (xmax - xmin) / (ymax - ymin);
}

/*------------------------------------------------*/
/*   Calculate the change velocity of ratio       */
/*------------------------------------------------*/
void change(float* tmpBufferRatio, int nb )
{
   for(int i=0; i<nb-1; i++){
     tmpBufferRatio[i] = tmpBufferRatio[i+1] - tmpBufferRatio[i]; 
   }  
}


/*-------------------------------------------------------------------------------------*/
/*        Compute PDF of Gaussian  provided by http://code.google.com/p/opencvx/       */
/*-------------------------------------------------------------------------------------*/
void cvMatGaussPdf( CvMat* samples, CvMat* mean, CvMat* cov, CvMat* probs, bool normalize CV_DEFAULT(true), bool logprob CV_DEFAULT(false) )
{
    int D = samples->rows;
    int N = samples->cols;
    int type = samples->type;
    
    CvMat *invcov = cvCreateMat( D, D, type );
    cvInvert( cov, invcov, CV_SVD );

    CvMat *sample = cvCreateMat( D, 1, type );
    CvMat *subsample   = cvCreateMat( D, 1, type );
    CvMat *subsample_T = cvCreateMat( 1, D, type );
    CvMat *value       = cvCreateMat( 1, 1, type );

    double prob;
    for( int n = 0; n < N; n++ )
    {
        cvGetCol( samples, sample, n );

        cvSub( sample, mean, subsample );
        cvTranspose( subsample, subsample_T );
        cvMatMul( subsample_T, invcov, subsample_T );
        cvMatMul( subsample_T, subsample, value );
        prob = -0.5 * cvmGet(value, 0, 0);
        if( !logprob ) prob = exp( prob );

        cvmSet( probs, 0, n, prob );
    }
    if( normalize )
    {
        double norm = pow( 2* M_PI, D/2.0 ) * sqrt( cvDet( cov ) );
        if( logprob ) cvSubS( probs, cvScalar( log( norm ) ), probs );
        else cvConvertScale( probs, probs, 1.0 / norm );
    }
    
    cvReleaseMat( &invcov );
    cvReleaseMat( &sample );
    cvReleaseMat( &subsample );
    cvReleaseMat( &subsample_T );
    cvReleaseMat( &value );

}


float cvGaussPdf( CvMat* sample, CvMat* mean, CvMat* cov, bool normalize CV_DEFAULT(true), bool logprob CV_DEFAULT(false))
{
    float prob;
    CvMat *_probs  = cvCreateMat( 1, 1, sample->type );

    cvMatGaussPdf( sample, mean, cov, _probs, normalize, logprob );
    prob = cvmGet(_probs, 0, 0);

    cvReleaseMat( &_probs );
    return prob;
}

/*-----------------------------------------------------------------------------------------------*/
/*        the statistics of data:   nb of frames of each scenes captured by one camera           */ 
/*                                  total nb of frames captured by one camera                    */
/*                                  the index of image begining and ending in each scenes        */
/*-----------------------------------------------------------------------------------------------*/
void statis(int start_scene, int end_scene, int* start_frames, int* end_frames, int* total_frames, int nb_cam, int &sum_nb )
{	
 
  int selectedFrames[1000];
  char fname[260];
  
  for(int i=0;i<1000;i++) selectedFrames[i]=i;  

  for(int i=start_scene; i<=end_scene; i++){
    
    IplImage* tmp;
    int nb = 0;
    do{
        nb++;
		sprintf(fname,"%02d/cam%01d/%03d.jpg",i,nb_cam,selectedFrames[nb]);
        tmp = cvLoadImage(fname,0);
             
    }while(tmp!=NULL);

    cvReleaseImage(&tmp);

    end_frames[i] = nb - 40;
    total_frames[i] = end_frames[i] - start_frames[i] ;

  }
  
  
  for(int i=0; i<=end_scene; i++){
    sum_nb = sum_nb + total_frames[i];
  }

}

/*-----------------------------------------------------------*/
/*        Load all the data captured by each camera          */
/*-----------------------------------------------------------*/

void loadData (int start_scene, int end_scene, int* start_bg, int* end_bg, int* start_frames, int* end_frames, 
                               int* total_frames, float* label, int nb_cam, int sum_nb, point* frame )
{ 

  
  int key = 0;
  int accum = 0;
  const CvSize sz= cvSize(320,240);
  float xmax, xmin, ymax, ymin;
  float ratioBuffer[sum_nb];
  float changBuffer[sum_nb];
  
  int selectedFrames[1000];
  char fname[260];
  
  for(int i=0;i<1000;i++) selectedFrames[i]=i; 

  for(int k=start_scene; k<=end_scene; k++){
    int nbEtiq;
    IplImage* background = bg(k, nb_cam, start_bg[k], end_bg[k]);    
    
    float tmpBufferRatio[1000];
   
    for(int j=0; j<1000; j++) tmpBufferRatio[j] = 0;
     
    for(int n=start_frames[k]; n<end_frames[k]; n++){
      sprintf(fname,"%02d/cam%01d/%03d.jpg",k,nb_cam,selectedFrames[n]); 
      IplImage* img = cvLoadImage(fname,0); 
      IplImage* segmentation = cvCreateImage(sz,IPL_DEPTH_8U,1); 
      IplImage* imgEtiq;
      CvMat* matEtiq;
      if(img){	          
        cvAbsDiff(background,img,segmentation);
        cvThreshold(segmentation,segmentation,15,255,CV_THRESH_BINARY);
        cvErode(segmentation,segmentation,0,2);
        cvDilate(segmentation,segmentation,0,9);
      
        nbEtiq = etiquetage(segmentation, matEtiq);
        convertEtiq(matEtiq, imgEtiq, nbEtiq);
        select(imgEtiq, xmax, xmin, ymax, ymin);
 
        tmpBufferRatio[n-start_frames[k]] = ratio1(xmax,xmin,ymax,ymin);
           
      }	 
 
      cvReleaseImage(&img);
      cvReleaseImage(&imgEtiq);	
      cvReleaseImage(&segmentation);	
      cvReleaseMat(&matEtiq);
              
    }
  
    
    accum = accum + total_frames[k-1];    

    for(int j=0; j<total_frames[k]; j++){
      ratioBuffer[j+accum] = tmpBufferRatio[j];
    }

    change(tmpBufferRatio,total_frames[k]);
    
    for(int j=0; j<total_frames[k]; j++){
      changBuffer[j+accum] = tmpBufferRatio[j];
    }  

    cvReleaseImage(&background);
   
  }  

  for(int i=0; i<sum_nb; i++){
      frame[i].ratio = ratioBuffer[i];
      frame[i].vitess = changBuffer[i]; 
  }

}  


/*----------------------------------------------------------*/
/*        Estimate parametres of mixtured Gaussian          */
/*----------------------------------------------------------*/

void paramEstim (point* frames, point &mean0, point &mean1, 
                    Mat var0, Mat var1, float &p0, float &p1, float* label, int sum_nb )
{	

  /*------------------k-means------------------------------*/
 	  
  mean0.ratio = 3,  mean0.vitess = 0.05;
  mean1.ratio = 0.4,  mean1.vitess = 0.01;
  
  point sum0, sum1;
  point l, k;

  int count0, count1;

  do{	

     sum0.ratio = 0, sum0.vitess = 0; 
     sum1.ratio = 0, sum1.vitess = 0;
     count0 = 0 , count1 = 0;
     
     l = mean0 , k = mean1; 

     for(int i=0; i<sum_nb; i++){
		if(sqrt(pow(frames[i].ratio-mean0.ratio,2)+pow(frames[i].vitess-mean0.vitess,2)) < sqrt(pow(frames[i].ratio-mean1.ratio,2)+pow(frames[i].vitess-mean1.vitess,2))){
	        label[i] = 0;    
	    }
	    else{
	        label[i] = 1;
	    }
     }					
     
     for(int i=0; i<sum_nb; i++){
		if(label[i] == 0){
		   	sum0.ratio = sum0.ratio + frames[i].ratio;
	        sum0.vitess = sum0.vitess + frames[i].vitess;
	        count0++;
	    }
	    else{
	        sum1.ratio = sum1.ratio + frames[i].ratio;
	        sum1.vitess = sum1.vitess + frames[i].vitess;
	        count1++;
	    }			
     }			
  

     mean0.ratio = sum0.ratio/count0;
     mean0.vitess = sum0.vitess/count0;
     mean1.ratio = sum1.ratio/count1;
     mean1.vitess = sum1.vitess/count1;    

     
  }while(mean0.ratio!=l.ratio||mean0.vitess!=l.vitess||
            mean1.ratio!=k.ratio||mean1.vitess!=k.vitess);


  Mat temp0 = Mat(2,2,CV_32F); 
  Mat temp1 = Mat(2,2,CV_32F);

  for(int i=0; i<sum_nb; i++){
     if(label[i] == 0){
       temp0.at<float>(0,0) = frames[i].ratio - mean0.ratio;
       temp0.at<float>(1,0) = frames[i].vitess - mean0.vitess;
       temp0.at<float>(0,1) = 0;
       temp0.at<float>(1,1) = 0;
       var0 = var0 + temp0 * temp0.t();
     }else{
       temp1.at<float>(0,0) = frames[i].ratio - mean1.ratio;
       temp1.at<float>(1,0) = frames[i].vitess - mean1.vitess;
       temp1.at<float>(0,1) = 0;
       temp1.at<float>(1,1) = 0;
       var1 = var1 + temp1 * temp1.t();
     }
  }

  var0 = var0 / count0;
  var1 = var1 / count1;
 
  p0 = (float)count0 / (count0 + count1);
  p1 = 1-p0;


  /*--------------------------EM algorithem------------------------------------*/  
  
  float karma_n0, karma_n1;
  float prob0, prob1, prob0new, prob1new;
  float sum, sum_new;  
  float n0,n1;
  float p0old, p1old;
 
  point mean0old, mean1old;
    
  Mat var0old;
  Mat var1old;
 
  do{
 
    mean0old = mean0;
    mean1old = mean1;
    var0old = var0;
    var1old = var1;
    p0old = p0;
    p1old = p1;

    sum = 0, sum_new = 0;
    n0 = 0, n1 = 0;
    sum0.ratio = 0, sum0.vitess = 0;
    sum1.ratio = 0, sum1.vitess = 0;    

    for(int n=0; n<sum_nb; n++){

       float v[] = { frames[n].ratio, frames[n].vitess };
       float m0[] = { mean0old.ratio, mean0old.vitess };
       float m1[] = { mean1old.ratio, mean1old.vitess };
       float a0[] = { var0old.at<float>(0,0),var0old.at<float>(0,1), 
                        var0old.at<float>(1,0), var0old.at<float>(1,1) };
       float a1[] = { var1old.at<float>(0,0),var1old.at<float>(0,1), 
                        var1old.at<float>(1,0), var1old.at<float>(1,1) };
      
       CvMat vec = cvMat(2, 1, CV_32F, v);
       CvMat mean_0 = cvMat(2, 1, CV_32F, m0);
       CvMat mean_1 = cvMat(2, 1, CV_32F, m1);
       CvMat cov_0 = cvMat(2, 2, CV_32F, a0);
       CvMat cov_1 = cvMat(2, 2, CV_32F, a1);
       
       prob0 = cvGaussPdf( &vec, &mean_0, &cov_0);
       prob1 = cvGaussPdf( &vec, &mean_1, &cov_1);
          
       karma_n0 = p0old*prob0 / (p0old*prob0+p1old*prob1);
       karma_n1 = p1old*prob1 / (p0old*prob0+p1old*prob1);

       n0 = n0 + karma_n0;
       n1 = n1 + karma_n1;

       sum0.ratio = sum0.ratio + karma_n0 * frames[n].ratio;
       sum0.vitess = sum0.vitess + karma_n0 * frames[n].vitess;
       sum1.ratio = sum1.ratio + karma_n1 * frames[n].ratio;
       sum1.vitess = sum1.vitess + karma_n1 * frames[n].vitess;

    }
  
    mean0.ratio = sum0.ratio / n0;
    mean0.vitess = sum0.vitess / n0;
    mean1.ratio = sum1.ratio / n1;
    mean1.vitess = sum1.vitess / n1;

    Mat var0new = Mat::zeros(2,2,CV_32F);
    Mat var1new = Mat::zeros(2,2,CV_32F);

    for(int n=0; n<sum_nb; n++){

       float v[] = { frames[n].ratio, frames[n].vitess };
       float m0[] = { mean0old.ratio, mean0old.vitess };
       float m1[] = { mean1old.ratio, mean1old.vitess };
       float a0[] = { var0old.at<float>(0,0),var0old.at<float>(0,1), 
                        var0old.at<float>(1,0), var0old.at<float>(1,1) };
       float a1[] = { var1old.at<float>(0,0),var1old.at<float>(0,1), 
                        var1old.at<float>(1,0), var1old.at<float>(1,1) };

       CvMat vec  = cvMat(2, 1, CV_32F, v);
       CvMat mean_0 = cvMat(2, 1, CV_32F, m0);
       CvMat mean_1 = cvMat(2, 1, CV_32F, m1);
       CvMat cov_0  = cvMat(2, 2, CV_32F, a0);
       CvMat cov_1  = cvMat(2, 2, CV_32F, a1);

       prob0 = cvGaussPdf( &vec, &mean_0, &cov_0);
       prob1 = cvGaussPdf( &vec, &mean_1, &cov_1);
        
       karma_n0 = p0old*prob0 / (p0old*prob0+p1old*prob1);
       karma_n1 = p1old*prob1 / (p0old*prob0+p1old*prob1);

       temp0.at<float>(0,0) = frames[n].ratio - mean0.ratio;
       temp0.at<float>(1,0) = frames[n].vitess - mean0.vitess;
       temp0.at<float>(0,1) = 0;
       temp0.at<float>(1,1) = 0;

       temp1.at<float>(0,0) = frames[n].ratio - mean1.ratio;
       temp1.at<float>(1,0) = frames[n].vitess - mean1.vitess;
       temp1.at<float>(0,1) = 0;
       temp1.at<float>(1,1) = 0;
   
       var0new = var0new + karma_n0*temp0*temp0.t();
       var1new = var1new + karma_n1*temp1*temp1.t();
   
    }

    var0 = var0new / n0;
    var1 = var1new / n1;

    p0 = n0 / (n0+n1);
    p1 = n1 / (n0+n1);


    for(int n=0; n<sum_nb; n++){
 
      float v[] = { frames[n].ratio, frames[n].vitess };
      float m0old[] = { mean0old.ratio, mean0old.vitess };
      float m1old[] = { mean1old.ratio, mean1old.vitess };
      float a0old[] = { var0old.at<float>(0,0),var0old.at<float>(0,1), 
                           var0old.at<float>(1,0), var0old.at<float>(1,1) };
      float a1old[] = { var1old.at<float>(0,0),var1old.at<float>(0,1), 
                           var1old.at<float>(1,0), var1old.at<float>(1,1) };

      CvMat vec = cvMat(2, 1, CV_32F, v);
      CvMat mean_0_old = cvMat(2, 1, CV_32F, m0old);
      CvMat mean_1_old = cvMat(2, 1, CV_32F, m1old);
      CvMat cov_0_old  = cvMat(2, 2, CV_32F, a0old);
      CvMat cov_1_old  = cvMat(2, 2, CV_32F, a1old);

      prob0 = cvGaussPdf( &vec, &mean_0_old, &cov_0_old);
      prob1 = cvGaussPdf( &vec, &mean_1_old, &cov_1_old);

      float m0[] = { mean0.ratio, mean0.vitess };
      float m1[] = { mean1.ratio, mean1.vitess };
      float a0[] = { var0.at<float>(0,0),var0.at<float>(0,1), 
                          var0.at<float>(1,0), var0.at<float>(1,1) };
      float a1[] = { var1.at<float>(0,0),var1.at<float>(0,1), 
                          var1.at<float>(1,0), var1.at<float>(1,1) };

      CvMat mean_0 = cvMat(2, 1, CV_32F, m0);
      CvMat mean_1 = cvMat(2, 1, CV_32F, m1);
      CvMat cov_0  = cvMat(2, 2, CV_32F, a0);
      CvMat cov_1  = cvMat(2, 2, CV_32F, a1);

      prob0new = cvGaussPdf( &vec, &mean_0, &cov_0);
      prob1new = cvGaussPdf( &vec, &mean_1, &cov_1);
       
      sum = sum + log(p0old*prob0 + p1old*prob1);
      sum_new = sum_new + log(p0*prob0new + p1*prob1new);
       
    }  
    
    
  }while(sum!=sum_new);
   
}

/*----------------------------------------------------------*/
/*                Training Cameras                          */
/*----------------------------------------------------------*/
void TrainingCam1(point &mean0, point &mean1, Mat var0, Mat var1, float &p0, float &p1)
{ 
    int start_frames[11] = {0, 189, 75, 90, 70, 72, 97, 78, 51, 75, 74};
    int cam_start_bg[11] = {0, 10, 10, 20, 10, 10, 25, 10, 10, 10, 11};
    int cam_end_bg[11] = {0, 170, 67, 80, 60, 60, 90, 69, 40, 68, 69};
 
    int sum_nb = 0;
    int end_frames[11] = {};
    int total_frames[11] = {};

    statis(1, 10, start_frames, end_frames, total_frames, 1, sum_nb);

    float label[sum_nb];
    for(int i=0; i<sum_nb; i++) label[i] = 0;

    point* frames = (point *) malloc (sum_nb*sizeof(point));
    for(int i=0; i<sum_nb; i++) frames[i].ratio = 0.0;
    for(int i=0; i<sum_nb; i++) frames[i].vitess = 0.0;

    loadData(1, 10, cam_start_bg, cam_end_bg, start_frames, 
                       end_frames, total_frames, label, 1, sum_nb, frames);
    
	// printf("%f\n", mean0);


    paramEstim(frames, mean0, mean1, var0, var1, p0, p1, label, sum_nb); 
    
}	

void TrainingCam2(point &mean0, point &mean1, Mat var0, Mat var1, float &p0, float &p1)
{ 
    int start_frames[11] = {0, 189, 70, 82, 48, 62, 88, 68, 48, 68, 72};
    int cam_start_bg[11] = {0, 20, 10, 26, 10, 10, 30, 12, 10, 15, 23};
    int cam_end_bg[11] = {0, 170, 60, 74, 38, 50, 78, 60, 37, 60, 64};
 
    int sum_nb = 0;
    int end_frames[11] = {};
    int total_frames[11] = {};
    
    statis(1, 10, start_frames, end_frames, total_frames, 2, sum_nb);
     
    float label[sum_nb];
    for(int i=0; i<sum_nb; i++) label[i] = 0;
   
    point* frames = (point *) malloc (sum_nb*sizeof(point));
    for(int i=0; i<sum_nb; i++) frames[i].ratio = 0;
    for(int i=0; i<sum_nb; i++) frames[i].vitess = 0;

    loadData(1, 10, cam_start_bg, cam_end_bg, start_frames, 
                       end_frames, total_frames, label, 2, sum_nb, frames);
      
    paramEstim(frames, mean0, mean1, var0, var1, p0, p1, label, sum_nb); 
    
}	

void TrainingCam3(point &mean0, point &mean1, Mat var0, Mat var1, float &p0, float &p1)
{ 
    int start_frames[11] = {0, 188, 85, 86, 77, 71, 100, 89, 57, 69, 80};
    int cam_start_bg[11] = {0, 25, 20, 30, 15, 15, 30, 15, 10, 14, 21};
    int cam_end_bg[11] = {0, 170, 60, 70, 54, 55, 82, 63, 38, 60, 60};
 
    int sum_nb = 0;
    int end_frames[11] = {};
    int total_frames[11] = {};

    statis(1, 10, start_frames, end_frames, total_frames, 3, sum_nb);

    float label[sum_nb];
    for(int i=0; i<sum_nb; i++) label[i] = 0;

    point* frames = (point *) malloc (sum_nb*sizeof(point));
    for(int i=0; i<sum_nb; i++) frames[i].ratio = 0;
    for(int i=0; i<sum_nb; i++) frames[i].vitess = 0;
 
    loadData(1, 10, cam_start_bg, cam_end_bg, start_frames, 
                       end_frames, total_frames, label, 3, sum_nb, frames);
        
    paramEstim(frames, mean0, mean1, var0, var1, p0, p1, label, sum_nb); 
     
}	

void TrainingCam4(point &mean0, point &mean1, Mat var0, Mat var1, float &p0, float &p1)
{ 
    int start_frames[11] = {0, 185, 62, 81, 59, 58, 82, 66, 39, 65, 67};
    int cam_start_bg[11] = {0, 120, 23, 47, 20, 22, 45, 28, 18, 25, 34};
    int cam_end_bg[11] = {0, 169, 47, 68, 44, 44, 70, 50, 27, 51, 47};
 
    int sum_nb = 0;
    int end_frames[11] = {};
    int total_frames[11] = {};

    statis(1, 10, start_frames, end_frames, total_frames, 4, sum_nb);

    float label[sum_nb];
    for(int i=0; i<sum_nb; i++) label[i] = 0;

    point* frames = (point *) malloc (sum_nb*sizeof(point));
    for(int i=0; i<sum_nb; i++) frames[i].ratio = 0;
    for(int i=0; i<sum_nb; i++) frames[i].vitess = 0;
     
    loadData(1, 10, cam_start_bg, cam_end_bg, start_frames, 
                       end_frames, total_frames, label, 4, sum_nb, frames);
     
    paramEstim(frames, mean0, mean1, var0, var1, p0, p1, label, sum_nb); 
    
    
}	





