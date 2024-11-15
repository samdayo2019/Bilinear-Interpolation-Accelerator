#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define B 1 // Batch size
#define C 3 // Input image channel dimensions (RGB)
#define H 192 // Image height
#define W 320 // Image width 

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
            int r = (int)(image[0][0][i][j]);
            int g = (int)(image[0][1][i][j]);
            int b = (int)(image[0][2][i][j]);

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
    int i, j, b, k, c; 

    // Build the tensor of sampling coordinates p_s ~ K*T*D*K^(-1)*p_t
    for(b = 0; b < B; b++){
        inverse(K, K_inv, b);

        // Initialize p_t and Dt_flat
        for(j = 0; j < H * W; j++){
            int y = j / W; // Row index
            int x = j % W; // Column index
            p_t[b][0][j] = x; // x-coordinate
            p_t[b][1][j] = y; // y-coordinate
            p_t[b][2][j] = 1.0f;
            Dt_flat[b][0][j] = Dt[b][0][y][x];
            cam_coords[b][3][j] = 1.0f; // Homogeneous coordinate
        }

        // Compute cam_coords = Dt * K_inv * p_t
        for(i = 0; i < 3; i++){
            for (j = 0; j < H * W; j++){
                float sum = 0.0f;
                for (k = 0; k < 3; k++){
                    sum += K_inv[b][i][k] * p_t[b][k][j];
                }
                cam_coords[b][i][j] = sum * Dt_flat[b][0][j];
            }
        }

        // Compute src_coords = T * cam_coords
        for(j = 0; j < H * W; j++){
            float cam_coord[4];
            for (i = 0; i < 4; i++) {
                cam_coord[i] = cam_coords[b][i][j];
            }

            for(i = 0; i < 3; i++){
                float sum = 0.0f;
                for (k = 0; k < 4; k++){
                    sum += T[b][i][k] * cam_coord[k];
                }
                src_coords[b][i][j] = sum;
            }
        }

        // Compute p_s_flat = K * src_coords
        for(i = 0; i < 3; i++){
            for (j = 0; j < H * W; j++){
                float sum = 0.0f;
                for (k = 0; k < 3; k++){
                    sum += K[b][i][k] * src_coords[b][k][j];
                }
                p_s_flat[b][i][j] = sum;
            }
        }

        // Normalize to get pixel coordinates
        for(j = 0; j < H * W; j++){
            int y = j / W;
            int x = j % W;
            float x_s = p_s_flat[b][0][j] / p_s_flat[b][2][j];
            float y_s = p_s_flat[b][1][j] / p_s_flat[b][2][j];

            p_s[b][0][y][x] = x_s; // x-coordinate
            p_s[b][1][y][x] = y_s; // y-coordinate
        }

        // Perform bilinear sampling
        for(i = 0; i < H; i++){
            for(j = 0; j < W; j++){
                float x_s = p_s[b][0][i][j];
                float y_s = p_s[b][1][i][j];

                int x0 = (int)(x_s);
                int x1 = x0 + 1;
                int y0 = (int)(y_s);
                int y1 = y0 + 1;

                float x_frac = x_s - x0;
                float y_frac = y_s - y0;

                float w00 = (1 - x_frac) * (1 - y_frac); // Top-left
                float w01 = (1 - x_frac) * y_frac;       // Bottom-left
                float w10 = x_frac * (1 - y_frac);       // Top-right
                float w11 = x_frac * y_frac;             // Bottom-right

                for (c = 0; c < C; c++) {
                    float sum = 0.0f;

                    if (x0 >= 0 && x0 < W && y0 >= 0 && y0 < H)
                        sum += Is[b][c][y0][x0] * w00;

                    if (x0 >= 0 && x0 < W && y1 >= 0 && y1 < H)
                        sum += Is[b][c][y1][x0] * w01;

                    if (x1 >= 0 && x1 < W && y0 >= 0 && y0 < H)
                        sum += Is[b][c][y0][x1] * w10;

                    if (x1 >= 0 && x1 < W && y1 >= 0 && y1 < H)
                        sum += Is[b][c][y1][x1] * w11;

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
        for (int i = 0; i < H; i++) {
            for (int j = 0; j < W; j++) {
                Is[b][0][i][j] = 135;  // Red channel
                Is[b][1][i][j] = 206;  // Green channel
                Is[b][2][i][j] = 235;  // Blue channel
            }
        }

            // Draw a simple house
        int house_start_x = 80;
        int house_start_y = 120;
        int house_width = 100;
        int house_height = 80;

        for (int i = house_start_y; i < house_start_y + house_height; i++) {
            for (int j = house_start_x; j < house_start_x + house_width; j++) {
                Is[b][0][i][j] = 139;  // Red channel (brown)
                Is[b][1][i][j] = 69;   // Green channel
                Is[b][2][i][j] = 19;   // Blue channel
            }
        }

        // Draw the roof (red triangle)
        for (int i = 0; i < 40; i++) {
            for (int j = house_start_x - i; j <= house_start_x + house_width + i; j++) {
                if (j >= house_start_x && j <= house_start_x + house_width) {
                    Is[b][0][house_start_y - i][j] = 165;  // Red channel
                    Is[b][1][house_start_y - i][j] = 42;   // Green channel
                    Is[b][2][house_start_y - i][j] = 42;   // Blue channel
                }
            }
        }

        // Draw the door (simple rectangle)
        int door_start_x = house_start_x + house_width / 2 - 10;
        int door_start_y = house_start_y + house_height - 30;
        int door_width = 20;
        int door_height = 30;

        for (int i = door_start_y; i < door_start_y + door_height; i++) {
            for (int j = door_start_x; j < door_start_x + door_width; j++) {
                Is[b][0][i][j] = 101;  // Red channel (dark brown)
                Is[b][1][i][j] = 67;   // Green channel
                Is[b][2][i][j] = 33;   // Blue channel
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
        T_t2s[b][0][0] = 0.5*cos(90); 
        T_t2s[b][0][1] = -1.1*sin(90);
        T_t2s[b][1][0] = 0.5*sin(90); 
        T_t2s[b][1][1] = -1.1*cos(90);
        T_t2s[b][0][3] = 0.02f; 
        T_t2s[b][1][3] = -0.03f; // Translate 0.1 units along x-axis
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






