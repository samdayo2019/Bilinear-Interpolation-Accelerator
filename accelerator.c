#include <stdio.h>
#include <stdlib.h>

#define B 1 // Batch size
#define C 3 // Input image channel dimensions (RGB)
#define H 192 // Image height
#define W 640 // Image width 

// Function prototypes
void image_reconstruction(float Is[B][C][H][W], float Dt[B][1][H][W], float T[B][4][4], float K[B][3][3], float It[B][C][H][W]);
void write_ppm(const char *filename, float image[B][C][H][W]);
void inverse(float input[B][3][3], float inv[B][3][3], int batch);


void inverse(float input[B][3][3], float inv[B][3][3], int batch){
    float det = input[batch][0][0] * (input[batch][1][1] * input[batch][2][2] - input[batch][1][2] * input[batch][2][1]) - 
        input[batch][0][1] * (input[batch][1][0] * input[batch][2][2] - input[batch][1][2] * input[batch][2][0]) +
        input[batch][0][2] * (input[batch][1][0] * input[batch][2][1] - input[batch][1][1] * input[batch][2][0]);
    
    float inv_det = 1.0f / det;

    inv[batch][0][0] = inv_det * (input[batch][1][1] * input[batch][2][2] - input[batch][1][2] * input[batch][2][1]);
    inv[batch][0][1] = inv_det * (input[batch][0][2] * input[batch][2][1] - input[batch][0][1] * input[batch][2][2]);
    inv[batch][0][2] = inv_det * (input[batch][0][1] * input[batch][1][2] - input[batch][0][2] * input[batch][1][1]);
    inv[batch][1][0] = inv_det * (input[batch][1][2] * input[batch][2][0] - input[batch][1][0] * input[batch][2][2]);
    inv[batch][1][1] = inv_det * (input[batch][0][0] * input[batch][2][2] - input[batch][0][2] * input[batch][2][0]);
    inv[batch][1][2] = inv_det * (input[batch][0][2] * input[batch][1][0] - input[batch][0][0] * input[batch][1][2]);
    inv[batch][2][0] = inv_det * (input[batch][1][0] * input[batch][2][1] - input[batch][1][1] * input[batch][2][0]);
    inv[batch][2][1] = inv_det * (input[batch][0][1] * input[batch][2][0] - input[batch][0][0] * input[batch][2][1]);
    inv[batch][2][2] = inv_det * (input[batch][0][0] * input[batch][1][1] - input[batch][0][1] * input[batch][1][0]);
}

// Function to write an image to a PPM file
void write_ppm(const char *filename, float image[B][C][H][W]) {
    FILE *fp = fopen(filename, "w");
    if (!fp) {
        printf("Unable to open file '%s' for writing.\n", filename);
        return;
    }

    // Write the PPM header
    fprintf(fp, "P3\n");
    fprintf(fp, "%d %d\n", W, H); // Width and Height
    fprintf(fp, "255\n"); // Max color value

    // Write pixel data
    for (int i = 0; i < H; i++) {
        for (int j = 0; j < W; j++) {
            // Since we're using B = 1, we access image[0]
            int r = (int)(image[0][0][i][j] * 255.0f);
            int g = (int)(image[0][1][i][j] * 255.0f);
            int b = (int)(image[0][2][i][j] * 255.0f);

            // Clamp values between 0 and 255
            if (r < 0) r = 0; if (r > 255) r = 255;
            if (g < 0) g = 0; if (g > 255) g = 255;
            if (b < 0) b = 0; if (b > 255) b = 255;

            fprintf(fp, "%d %d %d ", r, g, b);
        }
        fprintf(fp, "\n");
    }

    fclose(fp);
}

void image_reconstruction(float Is[B][C][H][W], float Dt[B][1][H][W], float T[B][4][4], float K[B][3][3], float It[B][C][H][W]){
    float Dt_flat[B][1][H * W];
    float p_t[B][3][H * W];
    float p_s_flat[B][3][H * W];
    float p_s[B][2][H][W];
    float K_inv[B][3][3];
    float cam_coords[B][4][H * W];
    float src_coords[B][3][H * W]; 
    int i, j, b, k, x, y, c, x_min, y_min; 
    float sum;
    float w_bi[B][4][H][W];
    float kernel[2][2];


    // build the tensor of sampling coordinates p_s ~K*T*D*K^(-1)*p_t

    // compute camera intrinsic matrix inverse
    // create the p_t tensor (each column holds homogenous coordinate -> x,y,1) 
    // flatten Dt into Dt_flat
    for(b = 0; b < B; b++){
        inverse(K, K_inv, b);
        for(j = 0; j < H * W; j++){
            x = j / W; 
            y = j % W;
            p_t[b][0][j] = x; 
            p_t[b][1][j] = y;
            p_t[b][2][j] = 1.0f;
            Dt_flat[b][0][j] = Dt[b][0][x][y];
            cam_coords[b][3][j] = 1.0f; // set up initially homogenous camera coordinate tensor
        }

        // compute Dt * K^(-1) * p_t
        for(i = 0; i < 3; i++){
            for (j = 0; j < H * W; j++){
                sum = 0.0f;
                for (k = 0; k < 3; k++){
                    sum += K_inv[b][i][k] * p_t[b][k][j];
                }
                cam_coords[b][i][j] = sum * Dt_flat[b][0][j];
            }
        }

        //compute T * Dt * K^(-1) * p_t ==> we only want the first 3 rows of the output, so we only compute using the first 3 rows of T
        for(i = 0; i < 3; i++){
            for (j = 0; j < H * W; j++){
                sum = 0.0f;
                for (k = 0; k < 4; k++){
                    sum += T[b][i][k] * cam_coords[b][k][j];
                }
                src_coords[b][i][j] = sum;
            }
        }

        //compute ps = K*T*D*K^(-1)*p_t
        for(i = 0; i < 3; i++){
            for (j = 0; j < H * W; j++){
                sum = 0.0f;
                for (k = 0; k < 3; k++){
                    sum += K[b][i][k] * src_coords[b][k][j];
                }
                p_s_flat[b][i][j] = sum;
            }
        }

        // normalize ps_flat to homogenous coordinates and stack into 2 channel tensor
        for (j = 0; j < H * W; j++){
            x = j / W; 
            y = j % W;
            p_s[b][0][x][y] = p_s_flat[b][0][j] / p_s_flat[b][2][j];
            p_s[b][1][x][y] = p_s_flat[b][1][j] / p_s_flat[b][2][j];
        }

        // Perform Bilinear Sampling using Is, p_s, and putting the result in It

        /*Step 1. take the x and y coordinates for the sampling location into two separate matrices
            Step 2. Find the floor, and the floor + 1 for each of the x and y coords, store then in x/y min/max matrices
            step 3. Compute weighted value matrices for bilinear sampling
            Step 4. Perform sampling by aligning 2x2 kernel with weighted values at x_min, y_min location and performing depth-wise 
                    sep conv with Is
            Step 5. Store final value for each channel at pixel coordinate holding sampling corrdinate in It. 
        */
        for(i = 0; i < H; i++){
            for(j = 0; j < W; j++){
                w_bi[b][0][i][j] = (p_s[b][0][i][j] - (int)p_s[b][0][i][j]) * (p_s[b][1][i][j] - (int)p_s[b][1][i][j]) ; // W00 = (x - floor(x))*(y - floor(y))
                w_bi[b][1][i][j] = (p_s[b][0][i][j] - (int)p_s[b][0][i][j]) * ((int)p_s[b][1][i][j] + 1 - p_s[b][1][i][j]) ; // W01 = (x - floor(x))*(floor(y) + 1 - y)
                w_bi[b][2][i][j] = ((int)p_s[b][0][i][j] + 1 - p_s[b][0][i][j]) * (p_s[b][1][i][j] - (int)p_s[b][1][i][j]) ; // W10 = (floor(x) + 1 - x)*(y - floor(y))
                w_bi[b][3][i][j] = ((int)p_s[b][0][i][j] + 1 - p_s[b][0][i][j]) * ((int)p_s[b][1][i][j] + 1 - p_s[b][1][i][j]) ; // W11 = (floor(x) + 1 - x)*(floor(y) + 1 - y)
            }
        }

        // perform bilinear sampling 
        for(i = 0; i < H; i++){
            for(j = 0; j < W; j++){
                // populate kernel
                kernel[0][0] = w_bi[b][0][i][j];
                kernel[0][1] = w_bi[b][1][i][j];
                kernel[1][0] = w_bi[b][2][i][j];
                kernel[1][1] = w_bi[b][3][i][j];
                // get sample x, y using xmin, ymin
                x_min = (int)p_s[b][0][i][j];
                y_min = (int)p_s[b][1][i][j];

                for(c = 0; c < 3; c++){                    
                    sum = 0;
                    // perform multiplications
                    for(x = 0; x < 2; x++){
                        for(y = 0; y < 2; y++){
                            sum += Is[b][c][x_min + x][y_min + y] * kernel[x][y];
                        }
                    }
                    It[b][c][i][j] = sum;                
                }
            }
        }
    }
}


int main() {
    // Define input tensors
    float Is[B][C][H][W];
    float Dt[B][1][H][W];
    float T_t2s[B][4][4];
    float K[B][3][3];
    float It_hat[B][C][H][W];

    int b, c, i, j;

    // Initialize Is (source image) with a gradient pattern
    for (b = 0; b < B; b++) {
        for (i = 0; i < H; i++) {
            for (j = 0; j < W; j++) {
                // Simple gradient pattern
                Is[b][0][i][j] = (float)j / (W - 1);       // Red channel
                Is[b][1][i][j] = (float)i / (H - 1);       // Green channel
                Is[b][2][i][j] = 0.5f;                     // Blue channel fixed at 0.5
            }
        }
    }

    // Initialize Dt (depth map) with a constant depth value
    for (b = 0; b < B; b++) {
        for (i = 0; i < H; i++) {
            for (j = 0; j < W; j++) {
                Dt[b][0][i][j] = 1.0f; // Depth of 1.0 units
            }
        }
    }

    // Initialize K (camera intrinsics matrix) with realistic values
    for (b = 0; b < B; b++) {
        float fx = 500.0f; // Focal length in pixels along x
        float fy = 500.0f; // Focal length in pixels along y
        float cx = (float)(W - 1) / 2.0f; // Principal point x-coordinate
        float cy = (float)(H - 1) / 2.0f; // Principal point y-coordinate

        K[b][0][0] = fx;   K[b][0][1] = 0.0f; K[b][0][2] = cx;
        K[b][1][0] = 0.0f; K[b][1][1] = fy;   K[b][1][2] = cy;
        K[b][2][0] = 0.0f; K[b][2][1] = 0.0f; K[b][2][2] = 1.0f;
    }

    // Initialize T_t2s (relative pose matrix) with a small translation along x
    for (b = 0; b < B; b++) {
        // Identity matrix
        for (i = 0; i < 4; i++)
            for (j = 0; j < 4; j++)
                T_t2s[b][i][j] = (i == j) ? 1.0f : 0.0f;

        // Apply a small translation
        T_t2s[b][0][3] = 0.1f; // Translate 0.1 units along x-axis
        // You can also add rotation by modifying the top-left 3x3 submatrix
    }

    // Call the reconstruct_image function
    image_reconstruction(Is, Dt, T_t2s, K, It_hat);

    // Write the source image and reconstructed image to PPM files
    write_ppm("source_image.ppm", Is);
    write_ppm("reconstructed_image.ppm", It_hat);

    printf("Images have been written to 'source_image.ppm' and 'reconstructed_image.ppm'.\n");

    return 0;
}






