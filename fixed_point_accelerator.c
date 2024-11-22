#include <stdint.h>
#include <stdio.h>
#include <math.h>
#include <corecrt_math_defines.h>

#define B 1        // Batch size
#define C 3        // Number of channels
#define H 256      // Image height
#define W 256      // Image width

#define FIXED_POINT_FRACTIONAL_BITS 16
#define FIXED_POINT_SCALE (1 << FIXED_POINT_FRACTIONAL_BITS)
#define FLOAT_TO_FIXED(x) ((int32_t)((x) * FIXED_POINT_SCALE))
#define FIXED_TO_FLOAT(x) ((float)(x) / FIXED_POINT_SCALE)
#define FIXED_MULT(a, b) ((int32_t)(((int64_t)(a) * (b)) >> FIXED_POINT_FRACTIONAL_BITS))
#define FIXED_DIV(a, b) ((int32_t)((((int64_t)(a) << FIXED_POINT_FRACTIONAL_BITS) / (b))))

void inverse_fixed_point(int32_t K[3][3], int32_t K_inv[3][3]);
void write_ppm(const char *filename, float image[B][C][H][W]);
void image_reconstruction_fixed_point();



void image_reconstruction_fixed_point(
    int8_t Is[B][C][H][W],
    int32_t Dt[B][1][H][W],
    int32_t T[B][4][4],
    int32_t K[B][3][3],
    int8_t It[B][C][H][W])
{
    int32_t Dt_flat[B][1][H * W];
    int32_t p_t[B][3][H * W];
    int32_t p_s_flat[B][3][H * W];
    int32_t p_s[B][2][H][W];
    int32_t K_inv[B][3][3];
    int32_t cam_coords[B][4][H * W];
    int32_t src_coords[B][3][H * W];
    int32_t w_bi[B][4][H][W];
    int32_t kernel[2][2];
    int i, j, b, k, x, y, c, x_min, y_min;
    int32_t sum;
    int64_t temp;

    for (b = 0; b < B; b++) {
        // Compute inverse of K in fixed-point
        inverse_fixed_point(K[b], K_inv[b]);

        // Flatten Dt into Dt_flat and set up p_t
        for (j = 0; j < H * W; j++) {
            x = j / W;
            y = j % W;
            p_t[b][0][j] = x << FIXED_POINT_FRACTIONAL_BITS;  // Fixed-point x
            p_t[b][1][j] = y << FIXED_POINT_FRACTIONAL_BITS;  // Fixed-point y
            p_t[b][2][j] = FLOAT_TO_FIXED(1.0f);
            Dt_flat[b][0][j] = Dt[b][0][x][y];
            cam_coords[b][3][j] = FLOAT_TO_FIXED(1.0f);
        }

        // Compute cam_coords = Dt * K_inv * p_t
        for (i = 0; i < 3; i++) {
            for (j = 0; j < H * W; j++) {
                sum = 0;
                for (k = 0; k < 3; k++) {
                    temp = FIXED_MULT(K_inv[i][k], p_t[b][k][j]);
                    sum += temp;
                }
                cam_coords[b][i][j] = FIXED_MULT(sum, Dt_flat[b][0][j]);
            }
        }

        // Transform cam_coords using T
        for (i = 0; i < 3; i++) {
            for (j = 0; j < H * W; j++) {
                sum = 0;
                for (k = 0; k < 4; k++) {
                    temp = FIXED_MULT(T[b][i][k], cam_coords[b][k][j]);
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
            p_s[b][0][x][y] = FIXED_DIV(p_s_flat[b][0][j], p_s_flat[b][2][j]);
            p_s[b][1][x][y] = FIXED_DIV(p_s_flat[b][1][j], p_s_flat[b][2][j]);
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

                int32_t one = FLOAT_TO_FIXED(1.0f);
                int32_t w00 = FIXED_MULT(one - dx, one - dy);
                int32_t w01 = FIXED_MULT(one - dx, dy);
                int32_t w10 = FIXED_MULT(dx, one - dy);
                int32_t w11 = FIXED_MULT(dx, dy);

                x_min = x0;
                y_min = y0;

                // Check bounds
                if (x_min >= 0 && x_min + 1 < H && y_min >= 0 && y_min + 1 < W) {
                    for (c = 0; c < C; c++) {
                        sum = 0;
                        for (x = 0; x < 2; x++) {
                            for (y = 0; y < 2; y++) {
                                int32_t I_val = Is[b][c][x_min + x][y_min + y];
                                I_val <<= FIXED_POINT_FRACTIONAL_BITS;
                                temp = FIXED_MULT(I_val, (x == 0 ? (y == 0 ? w00 : w01) : (y == 0 ? w10 : w11)));
                                sum += temp;
                            }
                        }
                        int32_t result = sum >> FIXED_POINT_FRACTIONAL_BITS;
                        if (result > 127) result = 127;
                        if (result < -128) result = -128;
                        It[b][c][i][j] = (int8_t)result;
                    }
                } else {
                    // Assign default value if out of bounds
                    for (c = 0; c < C; c++) {
                        It[b][c][i][j] = 0;
                    }
                }
            }
        }
    }
}

void inverse_fixed_point(int32_t K[3][3], int32_t K_inv[3][3]) {
    int64_t det_temp =
          (int64_t)K[0][0] * ((int64_t)K[1][1] * K[2][2] - (int64_t)K[1][2] * K[2][1])
        - (int64_t)K[0][1] * ((int64_t)K[1][0] * K[2][2] - (int64_t)K[1][2] * K[2][0])
        + (int64_t)K[0][2] * ((int64_t)K[1][0] * K[2][1] - (int64_t)K[1][1] * K[2][0]);
    int32_t det = (int32_t)(det_temp >> FIXED_POINT_FRACTIONAL_BITS);

    if (det != 0) {
        int32_t adj[3][3];
        adj[0][0] = FIXED_DIV(((int64_t)K[1][1] * K[2][2] - (int64_t)K[1][2] * K[2][1]) >> FIXED_POINT_FRACTIONAL_BITS, det);
        adj[0][1] = -FIXED_DIV(((int64_t)K[0][1] * K[2][2] - (int64_t)K[0][2] * K[2][1]) >> FIXED_POINT_FRACTIONAL_BITS, det);
        adj[0][2] = FIXED_DIV(((int64_t)K[0][1] * K[1][2] - (int64_t)K[0][2] * K[1][1]) >> FIXED_POINT_FRACTIONAL_BITS, det);
        adj[1][0] = -FIXED_DIV(((int64_t)K[1][0] * K[2][2] - (int64_t)K[1][2] * K[2][0]) >> FIXED_POINT_FRACTIONAL_BITS, det);
        adj[1][1] = FIXED_DIV(((int64_t)K[0][0] * K[2][2] - (int64_t)K[0][2] * K[2][0]) >> FIXED_POINT_FRACTIONAL_BITS, det);
        adj[1][2] = -FIXED_DIV(((int64_t)K[0][0] * K[1][2] - (int64_t)K[0][2] * K[1][0]) >> FIXED_POINT_FRACTIONAL_BITS, det);
        adj[2][0] = FIXED_DIV(((int64_t)K[1][0] * K[2][1] - (int64_t)K[1][1] * K[2][0]) >> FIXED_POINT_FRACTIONAL_BITS, det);
        adj[2][1] = -FIXED_DIV(((int64_t)K[0][0] * K[2][1] - (int64_t)K[0][1] * K[2][0]) >> FIXED_POINT_FRACTIONAL_BITS, det);
        adj[2][2] = FIXED_DIV(((int64_t)K[0][0] * K[1][1] - (int64_t)K[0][1] * K[1][0]) >> FIXED_POINT_FRACTIONAL_BITS, det);

        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                K_inv[i][j] = adj[i][j];
            }
        }
    } else {
        // Handle singular matrix
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                K_inv[i][j] = (i == j) ? FLOAT_TO_FIXED(1.0f) : 0;
            }
        }
    }
}
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


int main() {
    int8_t Is[B][C][H][W];
    int8_t It[B][C][H][W];
    int32_t Dt[B][1][H][W];
    int32_t K[B][3][3];
    int32_t T[B][4][4];

    // Initialize Is with a sample pattern
    for (int b = 0; b < B; b++) {
        for (int c = 0; c < C; c++) {
            for (int i = 0; i < H; i++) {
                for (int j = 0; j < W; j++) {
                    float value = 127.0f * sinf((float)(i + j) / 20.0f);
                    if (value > 127.0f) value = 127.0f;
                    if (value < -128.0f) value = -128.0f;
                    Is[b][c][i][j] = (int8_t)value;
                }
            }
        }
    }

    // Initialize Dt with varying depth values
    for (int b = 0; b < B; b++) {
        for (int i = 0; i < H; i++) {
            for (int j = 0; j < W; j++) {
                float depth = 1.0f + 0.5f * sinf((float)i / 50.0f) * cosf((float)j / 50.0f);
                Dt[b][0][i][j] = FLOAT_TO_FIXED(depth);
            }
        }
    }

    // Initialize K with realistic camera intrinsics
    for (int b = 0; b < B; b++) {
        float fx = 800.0f;
        float fy = 800.0f;
        float cx = (float)(W / 2);
        float cy = (float)(H / 2);

        K[b][0][0] = FLOAT_TO_FIXED(fx); K[b][0][1] = 0;             K[b][0][2] = FLOAT_TO_FIXED(cx);
        K[b][1][0] = 0;             K[b][1][1] = FLOAT_TO_FIXED(fy); K[b][1][2] = FLOAT_TO_FIXED(cy);
        K[b][2][0] = 0;             K[b][2][1] = 0;             K[b][2][2] = FLOAT_TO_FIXED(1.0f);
    }

    // Initialize T with a rotation and translation
    for (int b = 0; b < B; b++) {
        float angle = 10.0f * M_PI / 180.0f;  // Convert degrees to radians
        float tx = 0.1f;
        float ty = 0.0f;
        float tz = 0.0f;

        float cos_angle = cosf(angle);
        float sin_angle = sinf(angle);

        T[b][0][0] = FLOAT_TO_FIXED(cos_angle);  T[b][0][1] = FLOAT_TO_FIXED(-sin_angle); T[b][0][2] = 0;             T[b][0][3] = FLOAT_TO_FIXED(tx);
        T[b][1][0] = FLOAT_TO_FIXED(sin_angle);  T[b][1][1] = FLOAT_TO_FIXED(cos_angle);  T[b][1][2] = 0;             T[b][1][3] = FLOAT_TO_FIXED(ty);
        T[b][2][0] = 0;             T[b][2][1] = 0;             T[b][2][2] = FLOAT_TO_FIXED(1.0f); T[b][2][3] = FLOAT_TO_FIXED(tz);
        T[b][3][0] = 0;             T[b][3][1] = 0;             T[b][3][2] = 0;             T[b][3][3] = FLOAT_TO_FIXED(1.0f);
    }

    // Call the image reconstruction function
    image_reconstruction_fixed_point(Is, Dt, T, K, It);

    // Output the reconstructed image for validation
    for (int b = 0; b < B; b++) {
        for (int c = 0; c < C; c++) {
            printf("Batch %d, Channel %d:\n", b, c);
            for (int i = 0; i < H; i += H / 16) {
                for (int j = 0; j < W; j += W / 16) {
                    printf("%4d ", It[b][c][i][j]);
                }
                printf("\n");
            }
            printf("\n");
        }
    }

        // Write the source image and reconstructed image to PPM files
    write_ppm("source_image.ppm", Is);
    write_ppm("reconstructed_image.ppm", It);

    printf("Images have been written to 'source_image.ppm' and 'reconstructed_image.ppm'.\n");

    // Optionally, save It to a file or visualize it using an image library

    return 0;
}
