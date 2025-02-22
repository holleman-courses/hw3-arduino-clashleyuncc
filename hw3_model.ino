#include <TensorFlowLite.h>

#include <Arduino.h>
#include "sinModel2.h"
#include "tensorflow/lite/micro/tflite_bridge/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"
//#include "tensorflow/lite/version.h"

#define INPUT_BUFFER_SIZE 64
#define OUTPUT_BUFFER_SIZE 64
#define INT_ARRAY_SIZE 8

// put function declarations here:

namespace {
tflite::ErrorReporter* error_reporter = nullptr;
const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* model_input = nullptr;
TfLiteTensor* model_output = nullptr;
constexpr int kTensorArenaSize = 136 * 1024;
static uint8_t tensor_arena[kTensorArenaSize];
}

int string_to_array(char *in_str, int *int_array);
void print_int_array(int *int_array, int array_len);
int sum_array(int *int_array, int array_len);


char received_char = (char)NULL;              
int chars_avail = 0;                    // input present on terminal
char out_str_buff[OUTPUT_BUFFER_SIZE];  // strings to print to terminal
char in_str_buff[INPUT_BUFFER_SIZE];    // stores input from terminal
int input_array[INT_ARRAY_SIZE];        // array of integers input by user

int in_buff_idx=0; // tracks current input location in input buffer
int array_length=0;
int array_sum=0;

void setup() {
  delay(5000);
  static tflite::MicroMutableOpResolver<3> micro_op_resolver;
  micro_op_resolver.AddFullyConnected();
  micro_op_resolver.AddMul();
  micro_op_resolver.AddAdd();
  memset(in_str_buff, (char)0, INPUT_BUFFER_SIZE*sizeof(char));

  static tflite::MicroErrorReporter micro_error_reporter;
  error_reporter = &micro_error_reporter;
  model = tflite::GetModel(g_sinModel2);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    TF_LITE_REPORT_ERROR(error_reporter,
                         "Model provided is schema version %d not equal "
                         "to supported version %d.",
                         model->version(), TFLITE_SCHEMA_VERSION);
    return;
  }
  float t0 = millis();
  Serial.println("Timed Message");
  float t1 = millis();
  Serial.println("Starting Interpreter");
  static tflite::MicroInterpreter static_interpreter(
      model, micro_op_resolver, tensor_arena, kTensorArenaSize);
  interpreter = &static_interpreter;
  Serial.println("Interpreter done");
  model_input = interpreter->input(0);
  model_output = interpreter->output(0);
  // Allocate memory from the tensor_arena for the model's tensors.
  Serial.println("2");
  TfLiteStatus allocate_status = interpreter->AllocateTensors();
  if (allocate_status != kTfLiteOk) {
    TF_LITE_REPORT_ERROR(error_reporter, "AllocateTensors() failed");
    return;
  }

  // Get information about the memory area to use for the model's input.
  //model_input = interpreter->input(0);
  //model_output = interpreter->output(0);
  //Serial.println("Invoke starting");
  //TfLiteStatus invoke_status = interpreter->Invoke();
  //Serial.println("Invoke done");

  //if (invoke_status != kTfLiteOk) {
  //  TF_LITE_REPORT_ERROR(error_reporter, "Invoke failed");
  //  return;
  //}
  float t2 = millis();
  float t_print = t1 - t0;
  float t_infer = t2 - t1;
  Serial.print("Printing time: ");
  Serial.println(t_print);
  Serial.print("Inference time: ");
  Serial.println(t_infer);

  Serial.println("Enter seven comma seperated integers");
}

void loop() {
  // put your main code here, to run repeatedly:

  // check if characters are avialble on the terminal input
  chars_avail = Serial.available(); 
  if (chars_avail > 0) {
    received_char = Serial.read(); // get the typed character and 
    Serial.print(received_char);   // echo to the terminal

    in_str_buff[in_buff_idx++] = received_char; // add it to the buffer
    if (received_char == '\n') { // 13 decimal = newline character
      // user hit 'enter', so we'll process the line.
        if(in_buff_idx < 12) {
          Serial.println("Please enter 7 numbers");
        }
        else {
          Serial.print("About to process line: ");
          Serial.println(in_str_buff);
          array_length = string_to_array(in_str_buff, input_array);
          sprintf(out_str_buff, "Read in %d integers: ", array_length);
          Serial.print(out_str_buff);
          print_int_array(input_array, array_length);
          Serial.println("starting input loop");
          for (int i = 0; i < in_buff_idx; i++) {
            model_input->data.f[i] = input_array[i];
          }

          Serial.println("Starting interpreter");
          TfLiteStatus invoke_status = interpreter->Invoke();
          if (invoke_status != kTfLiteOk) {
            TF_LITE_REPORT_ERROR(error_reporter, "Invoke failed");
            return;
          }
          Serial.println("Starting output");
          TfLiteTensor* output = interpreter->output(0);
          int pred_val = model_output->data.f[0];
          Serial.print("Predicted Value: ");
          Serial.println(pred_val);
        }

      /*
      // Process and print out the array
      array_length = string_to_array(in_str_buff, input_array);
      sprintf(out_str_buff, "Read in  %d integers: ", array_length);
      Serial.print(out_str_buff);
      print_int_array(input_array, array_length);
      array_sum = sum_array(input_array, array_length);
      sprintf(out_str_buff, "Sums to %d\r\n", array_sum);
      Serial.print(out_str_buff);
      */
      // Now clear the input buffer and reset the index to 0
      memset(in_str_buff, (char)0, INPUT_BUFFER_SIZE*sizeof(char)); 
      in_buff_idx = 0;
    }
    else if (in_buff_idx >= INPUT_BUFFER_SIZE) {
      memset(in_str_buff, (char)0, INPUT_BUFFER_SIZE*sizeof(char)); 
      in_buff_idx = 0;
    }    
  }
}

int string_to_array(char *in_str, int *int_array) {
  int num_integers=0;
  char *token = strtok(in_str, ",");
  
  while (token != NULL) {
    int_array[num_integers++] = atoi(token);
    token = strtok(NULL, ",");
    if (num_integers >= INT_ARRAY_SIZE) {
      break;
    }
  }
  
  return num_integers;
}

void print_int_array(int *int_array, int array_len) {
  int curr_pos = 0; // track where in the output buffer we're writing

  sprintf(out_str_buff, "Integers: [");
  curr_pos = strlen(out_str_buff); // so the next write adds to the end
  for(int i=0;i<array_len;i++) {
    // sprintf returns number of char's written. use it to update current position
    curr_pos += sprintf(out_str_buff+curr_pos, "%d, ", int_array[i]);
  }
  sprintf(out_str_buff+curr_pos, "]\r\n");
  Serial.print(out_str_buff);
}

int sum_array(int *int_array, int array_len) {
  int curr_sum = 0; // running sum of the array

  for(int i=0;i<array_len;i++) {
    curr_sum += int_array[i];
  }
  return curr_sum;
}