#include <stdio.h>

int main() {
    printf("=== POINTER BASICS ===\n\n");
    
    // 1. Variables store values at memory addresses
    int x = 42;
    printf("Variable x = %d\n", x);
    printf("Address of x = %p\n", &x);  // & = "address of"
    
    // 2. Pointers store memory addresses
    int* ptr = &x;  // ptr points to x's address
    printf("Pointer ptr = %p (same as &x)\n", ptr);
    printf("Value at ptr = %d (same as x)\n", *ptr);  // * = "value at"
    
    printf("\n=== MODIFYING THROUGH POINTERS ===\n");
    
    /*// 3. Change value through pointer
    *ptr = 100;  // Changes x through the pointer
    printf("After *ptr = 100:\n");
    printf("x = %d (changed!)\n", x);
    printf("*ptr = %d\n", *ptr);
    
    printf("\n=== FUNCTION PARAMETERS ===\n");
    
    // 4. Function that takes value (copy)
    auto change_copy = [](int val) {
        val = 999;  // Only changes the copy
        printf("Inside function: val = %d\n", val);
    };
    
    // 5. Function that takes pointer (can modify original)
    auto change_original = [](int* ptr) {
        *ptr = 999;  // Changes the original variable
        printf("Inside function: *ptr = %d\n", *ptr);
    };
    
    x = 42;  // Reset x
    printf("Before change_copy: x = %d\n", x);
    change_copy(x);  // Pass value
    printf("After change_copy: x = %d (unchanged)\n", x);
    
    printf("\nBefore change_original: x = %d\n", x);
    change_original(&x);  // Pass address
    printf("After change_original: x = %d (changed!)\n", x);
    
    printf("\n=== ARRAYS AND POINTERS ===\n");
    
    // 6. Arrays are essentially pointers
    float arr[3] = {1.0f, 2.0f, 3.0f};
    float* arr_ptr = arr;  // Array name is a pointer to first element
    
    printf("arr[0] = %.1f\n", arr[0]);
    printf("*arr_ptr = %.1f (same as arr[0])\n", *arr_ptr);
    printf("arr_ptr[1] = %.1f (same as arr[1])\n", arr_ptr[1]);
    
    printf("\n=== CUDA CONTEXT ===\n");
    
    // 7. Why CUDA functions use pointers
    int device_count = 0;
    printf("Before: device_count = %d\n", device_count);
    
    // Simulate cudaGetDeviceCount behavior
    auto simulate_cuda_call = [](int* count) {
        *count = 2;  // "Found 2 devices"
    };
    
    simulate_cuda_call(&device_count);  // Must pass address
    printf("After: device_count = %d\n", device_count);*/
    
    return 0;
}