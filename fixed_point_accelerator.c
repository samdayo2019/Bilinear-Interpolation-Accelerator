#include <stdio.h>
#include <stdint.h>
#include <math.h>
#include <stdlib.h>

#define B 1        // Batch size
#define C 3        // Number of channels (RGB)
#define H 256      // Image height
#define W 256      // Image width

#define FIXED_POINT_FRACTIONAL_BITS 16
#define FIXED_POINT_SCALE (1 << FIXED_POINT_FRACTIONAL_BITS)
#define FLOAT_TO_FIXED(x) ((int32_t)((x) * FIXED_POINT_SCALE + ((x) >= 0 ? 0.5f : -0.5f)))
#define FIXED_TO_FLOAT(x) ((float)(x) / FIXED_POINT_SCALE)
#define FIXED_MULT(a, b) ((int32_t)(((int64_t)(a) * (b)) >> FIXED_POINT_FRACTIONAL_BITS))
#define FIXED_DIV(a, b) ((int32_t)((((int64_t)(a) << FIXED_POINT_FRACTIONAL_BITS) / (b))))

// Function prototypes
void inverse_fixed_point_upper_triangular(int32_t K[3][3], int32_t K_inv[3][3]);
void image_reconstruction_fixed_point(
    uint8_t Is[B][C][H][W],
    int32_t Dt[B][1][H][W],
    int32_t T_t2s[B][4][4],
    int32_t K[B][3][3],
    uint8_t It_hat[B][C][H][W]);
void write_ppm(const char* filename, uint8_t image[B][C][H][W]);

int main() {
    // Define input tensors
    uint8_t Is[B][C][H][W];
    uint8_t It_hat[B][C][H][W];
    int32_t Dt[B][1][H][W];
    int32_t T_t2s[B][4][4];
    int32_t K[B][3][3];
    int32_t K_inv[B][3][3];

    int b, c, i, j;

    // Initialize Is (source image) with a blue sky background
    for (b = 0; b < B; b++) {
        // Initialize all channels to 135 (Red), 206 (Green), 235 (Blue)
        for (c = 0; c < C; c++) {
            for (i = 0; i < H; i++) {
                for (j = 0; j < W; j++) {
                    Is[b][c][i][j] = 135;  // Red channel
                }
            }
        }
        for (c = 0; c < C; c++) {
            for (i = 0; i < H; i++) {
                for (j = 0; j < W; j++) {
                    Is[b][c][i][j] = (c == 1) ? 206 : (c == 2) ? 235 : Is[b][c][i][j];
                }
            }
        }

        // Draw a simple house
        int house_start_x = 80;
        int house_start_y = 120;
        int house_width = 100;
        int house_height = 80;

        for (i = house_start_y; i < house_start_y + house_height && i < H; i++) {
            for (j = house_start_x; j < house_start_x + house_width && j < W; j++) {
                Is[b][0][i][j] = 139;  // Red channel (brown)
                Is[b][1][i][j] = 69;   // Green channel
                Is[b][2][i][j] = 19;   // Blue channel
            }
        }

        // Draw the roof (red triangle)
        for (i = 0; i < 40; i++) {
            int y_coord = house_start_y - i;
            if (y_coord < 0) continue;
            for (j = house_start_x - i; j <= house_start_x + house_width + i; j++) {
                if (j >= 0 && j < W) {
                    Is[b][0][y_coord][j] = 165;  // Red channel
                    Is[b][1][y_coord][j] = 42;   // Green channel
                    Is[b][2][y_coord][j] = 42;   // Blue channel
                }
            }
        }

        // Draw the door (simple rectangle)
        int door_start_x = house_start_x + house_width / 2 - 10;
        int door_start_y = house_start_y + house_height - 30;
        int door_width = 20;
        int door_height = 30;

        for (i = door_start_y; i < door_start_y + door_height && i < H; i++) {
            for (j = door_start_x; j < door_start_x + door_width && j < W; j++) {
                Is[b][0][i][j] = 101;  // Red channel (dark brown)
                Is[b][1][i][j] = 67;   // Green channel
                Is[b][2][i][j] = 33;   // Blue channel
            }
        }
    }

    // Initialize Dt (depth map) with a constant depth value converted to fixed-point
    for (b = 0; b < B; b++) {
        for (i = 0; i < H; i++) {
            for (j = 0; j < W; j++) {
                Dt[b][0][i][j] = FLOAT_TO_FIXED(1.0f); // Depth of 1.0 units
            }
        }
    }

    // Initialize K (camera intrinsics matrix) with realistic values converted to fixed-point
    for (b = 0; b < B; b++) {
        float fx = 500.0f; // Focal length in pixels along x
        float fy = 500.0f; // Focal length in pixels along y
        float cx = (float)(W - 1) / 2.0f; // Principal point x-coordinate
        float cy = (float)(H - 1) / 2.0f; // Principal point y-coordinate

        K[b][0][0] = FLOAT_TO_FIXED(fx);   K[b][0][1] = 0;                   K[b][0][2] = FLOAT_TO_FIXED(cx);
        K[b][1][0] = 0;                   K[b][1][1] = FLOAT_TO_FIXED(fy);   K[b][1][2] = FLOAT_TO_FIXED(cy);
        K[b][2][0] = 0;                   K[b][2][1] = 0;                   K[b][2][2] = FLOAT_TO_FIXED(1.0f);
    }

    // Initialize K_inv using the specialized inversion function
    for (b = 0; b < B; b++) {
        inverse_fixed_point_upper_triangular(K[b], K_inv[b]);
    }

    // Initialize T_t2s (relative pose matrix) with identity transformation for verification
    for (b = 0; b < B; b++) {
        // Initialize as identity matrix (no rotation or translation)
        for (i = 0; i < 4; i++) {
            for (j = 0; j < 4; j++) {
                T_t2s[b][i][j] = (i == j) ? FLOAT_TO_FIXED(50.0f) : 0;
            }
        }
    }

    // Optional: Introduce a small rotation about the center
    /*
    for (b = 0; b < B; b++) {
        // Define rotation about the image center
        float angle_deg = 5.0f; // Rotate by 5 degrees
        float angle_rad = angle_deg * 3.14159265f / 180.0f;
        float cos_angle = cosf(angle_rad);
        float sin_angle = sinf(angle_rad);

        // Convert to fixed-point
        int32_t cos_fp = FLOAT_TO_FIXED(cos_angle);
        int32_t sin_fp = FLOAT_TO_FIXED(sin_angle);

        // Translation to origin
        T_t2s[b][0][3] = FLOAT_TO_FIXED(-((W - 1) / 2.0f));
        T_t2s[b][1][3] = FLOAT_TO_FIXED(-((H - 1) / 2.0f));

        // Rotation
        T_t2s[b][0][0] = cos_fp;
        T_t2s[b][0][1] = -sin_fp;
        T_t2s[b][1][0] = sin_fp;
        T_t2s[b][1][1] = cos_fp;

        // Translation back to center
        T_t2s[b][0][3] = FIXED_MULT(T_t2s[b][0][0], FLOAT_TO_FIXED((W - 1) / 2.0f)) +
                          FIXED_MULT(T_t2s[b][0][1], FLOAT_TO_FIXED((H - 1) / 2.0f));
        T_t2s[b][1][3] = FIXED_MULT(T_t2s[b][1][0], FLOAT_TO_FIXED((W - 1) / 2.0f)) +
                          FIXED_MULT(T_t2s[b][1][1], FLOAT_TO_FIXED((H - 1) / 2.0f));
    }
    */

    // Call the image reconstruction function
    image_reconstruction_fixed_point(Is, Dt, T_t2s, K, It_hat);

    // Write the source image and reconstructed image to PPM files
    write_ppm("source_image.ppm", Is);
    write_ppm("reconstructed_image.ppm", It_hat);

    printf("Images have been written to 'source_image.ppm' and 'reconstructed_image.ppm'.\n");

    return 0;
}

// Specialized inversion function for upper triangular matrix K
void inverse_fixed_point_upper_triangular(int32_t K[3][3], int32_t K_inv[3][3]) {
    // Extract floating-point values from fixed-point
    float fx = FIXED_TO_FLOAT(K[0][0]);
    float fy = FIXED_TO_FLOAT(K[1][1]);
    float cx = FIXED_TO_FLOAT(K[0][2]);
    float cy = FIXED_TO_FLOAT(K[1][2]);

    // Compute inverse values in floating-point
    float inv_fx = 1.0f / fx;
    float inv_fy = 1.0f / fy;
    float inv_cx_fx = -cx * inv_fx;
    float inv_cy_fy = -cy * inv_fy;

    // Convert inverse values back to fixed-point
    K_inv[0][0] = FLOAT_TO_FIXED(inv_fx);
    K_inv[0][1] = FLOAT_TO_FIXED(0.0f);
    K_inv[0][2] = FLOAT_TO_FIXED(inv_cx_fx);

    K_inv[1][0] = FLOAT_TO_FIXED(0.0f);
    K_inv[1][1] = FLOAT_TO_FIXED(inv_fy);
    K_inv[1][2] = FLOAT_TO_FIXED(inv_cy_fy);

    K_inv[2][0] = FLOAT_TO_FIXED(0.0f);
    K_inv[2][1] = FLOAT_TO_FIXED(0.0f);
    K_inv[2][2] = FLOAT_TO_FIXED(1.0f);
}

// Fixed-point image reconstruction function implementation
void image_reconstruction_fixed_point(
    uint8_t Is[B][C][H][W],
    int32_t Dt[B][1][H][W],
    int32_t T_t2s[B][4][4],
    int32_t K[B][3][3],
    uint8_t It_hat[B][C][H][W])
{
    // Temporary variables
    int32_t Dt_flat[B][1][H * W];
    int32_t p_t[B][3][H * W];
    int32_t p_s_flat[B][3][H * W];
    int32_t p_s[B][2][H][W];
    int32_t K_inv[B][3][3];
    int32_t cam_coords[B][4][H * W];
    int32_t src_coords[B][3][H * W];
    int i, j, b, k, x, y, c;
    int32_t sum;
    int64_t temp;

    for (b = 0; b < B; b++) {
        // Compute inverse of K in fixed-point using the specialized function
        // (Assuming K_inv has already been computed in main)
        // If not, uncomment the following line:
        inverse_fixed_point_upper_triangular(K[b], K_inv[b]);

        // Debug: Print K_inv for verification
        
        printf("K_inv Matrix for batch %d:\n", b);
        for(int m=0; m<3; m++) {
            for(int n=0; n<3; n++) {
                printf("%f ", FIXED_TO_FLOAT(K_inv[b][m][n]));
            }
            printf("\n");
        }
        printf("\n");
        

        // Flatten Dt into Dt_flat and set up p_t
        for (j = 0; j < H * W; j++) {
            y = j / W;
            x = j % W;
            p_t[b][0][j] = x << FIXED_POINT_FRACTIONAL_BITS;  // Fixed-point x
            p_t[b][1][j] = y << FIXED_POINT_FRACTIONAL_BITS;  // Fixed-point y
            p_t[b][2][j] = FLOAT_TO_FIXED(1.0f);             // Homogeneous coordinate
            Dt_flat[b][0][j] = Dt[b][0][y][x];
            cam_coords[b][3][j] = FLOAT_TO_FIXED(1.0f);       // Homogeneous coordinate
        }

        // Compute cam_coords = K_inv * p_t * Dt
        for (i = 0; i < 3; i++) {
            for (j = 0; j < H * W; j++) {
                sum = 0;
                for (k = 0; k < 3; k++) {
                    temp = FIXED_MULT(K_inv[b][i][k], p_t[b][k][j]);
                    sum += temp;
                }
                cam_coords[b][i][j] = FIXED_MULT(sum, Dt_flat[b][0][j]);
            }
        }

        // Transform cam_coords using T_t2s
        for (i = 0; i < 3; i++) {
            for (j = 0; j < H * W; j++) {
                sum = 0;
                for (k = 0; k < 4; k++) {
                    if (k == 3) {
                        // Multiply by homogeneous coordinate
                        temp = FIXED_MULT(T_t2s[b][i][k], FLOAT_TO_FIXED(1.0f));
                    } else {
                        temp = FIXED_MULT(T_t2s[b][i][k], cam_coords[b][k][j]);
                    }
                    sum += temp;
                }
                src_coords[b][i][j] = sum;
            }
        }

        // Compute p_s_flat = K * src_coords
        for (i = 0; i < 3; i++) {
            for (j = 0; j < H * W; j++) {
                sum = 0;
                for (k = 0; k < 3; k++) {
                    temp = FIXED_MULT(K[b][i][k], src_coords[b][k][j]);
                    sum += temp;
                }
                p_s_flat[b][i][j] = sum;
            }
        }

        // Normalize to homogeneous coordinates and stack into p_s
        for (j = 0; j < H * W; j++) {
            x = j / W;
            y = j % W;
            if (p_s_flat[b][2][j] != 0) {
                p_s[b][0][x][y] = FIXED_DIV(p_s_flat[b][0][j], p_s_flat[b][2][j]);
                p_s[b][1][x][y] = FIXED_DIV(p_s_flat[b][1][j], p_s_flat[b][2][j]);
            } else {
                p_s[b][0][x][y] = 0;
                p_s[b][1][x][y] = 0;
            }

            // Debug: Print p_s for specific pixels (e.g., center of the house)
            
            if (x == 85 && y == 123) {
                printf("p_s[%d][%d] = (%f, %f)\n", x, y, FIXED_TO_FLOAT(p_s[b][0][x][y]), FIXED_TO_FLOAT(p_s[b][1][x][y]));
            }
            
        }

        // Perform Bilinear Sampling
        for (i = 0; i < H; i++) {
            for (j = 0; j < W; j++) {
                int32_t x_s = p_s[b][0][i][j];
                int32_t y_s = p_s[b][1][i][j];

                int32_t x0 = x_s >> FIXED_POINT_FRACTIONAL_BITS;
                int32_t y0 = y_s >> FIXED_POINT_FRACTIONAL_BITS;

                int32_t dx = x_s - (x0 << FIXED_POINT_FRACTIONAL_BITS);
                int32_t dy = y_s - (y0 << FIXED_POINT_FRACTIONAL_BITS);

                // Clamp x0 and y0 to valid ranges
                if (x0 < 0) x0 = 0;
                if (x0 >= H - 1) x0 = H - 2;
                if (y0 < 0) y0 = 0;
                if (y0 >= W - 1) y0 = W - 2;

                // Compute weights
                int32_t one = FLOAT_TO_FIXED(1.0f);
                int32_t w00 = FIXED_MULT(one - dx, one - dy);
                int32_t w01 = FIXED_MULT(one - dx, dy);
                int32_t w10 = FIXED_MULT(dx, one - dy);
                int32_t w11 = FIXED_MULT(dx, dy);

                // Perform bilinear interpolation for each channel
                for (c = 0; c < C; c++) {
                    // Corrected mapping: Accessing [y0][x0] instead of [x0][y0]
                    int32_t I00 = ((int32_t)Is[b][c][y0][x0]) << FIXED_POINT_FRACTIONAL_BITS;
                    int32_t I01 = ((int32_t)Is[b][c][y0][x0 + 1]) << FIXED_POINT_FRACTIONAL_BITS;
                    int32_t I10 = ((int32_t)Is[b][c][y0 + 1][x0]) << FIXED_POINT_FRACTIONAL_BITS;
                    int32_t I11 = ((int32_t)Is[b][c][y0 + 1][x0 + 1]) << FIXED_POINT_FRACTIONAL_BITS;

                    // Apply weights
                    int32_t sum = FIXED_MULT(I00, w00)
                                + FIXED_MULT(I01, w01)
                                + FIXED_MULT(I10, w10)
                                + FIXED_MULT(I11, w11);

                    // Convert back to integer by shifting
                    int32_t result = sum >> FIXED_POINT_FRACTIONAL_BITS;

                    // Clamp the result to [0, 255]
                    if (result > 255) result = 255;
                    if (result < 0) result = 0;

                    It_hat[b][c][i][j] = (uint8_t)result;
                }
            }
        }
    }
}
// Function to write a PPM image (P6 format)
void write_ppm(const char* filename, uint8_t image[B][C][H][W]) {
    FILE* fp = fopen(filename, "wb");
    if (!fp) {
        printf("Failed to open file %s for writing.\n", filename);
        return;
    }

    // Write PPM header
    fprintf(fp, "P6\n%d %d\n255\n", W, H);

    // Write pixel data
    for (int b = 0; b < B; b++) {
        for (int i = 0; i < H; i++) {
            for (int j = 0; j < W; j++) {
                uint8_t pixel[3];
                pixel[0] = image[b][0][i][j]; // Red
                pixel[1] = image[b][1][i][j]; // Green
                pixel[2] = image[b][2][i][j]; // Blue
                fwrite(pixel, 1, 3, fp);
            }
        }
    }

    fclose(fp);
}
