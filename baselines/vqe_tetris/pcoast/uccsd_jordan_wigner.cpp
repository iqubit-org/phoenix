//===----------------------------------------------------------------------===//
// INTEL CONFIDENTIAL
//
// Copyright 2023 Intel Corporation.
//
// This software and the related documents are Intel copyrighted materials, and
// your use of them is governed by the express license under which they were
// provided to you ("License"). Unless the License provides otherwise, you may
// not use, modify, copy, publish, distribute, disclose or transmit this
// software or the related documents without Intel's prior written permission.
//
// This software and the related documents are provided as is, with no express
// or implied warranties, other than those that are expressly stated in the
// License.
//===----------------------------------------------------------------------===//
#include <clang/Quantum/quintrinsics.h>

/// Quantum Runtime Library APIs
#include <quantum.hpp>

const int total_qubits = 22;
const char* single_excitation[2] = {"XY", "YX"};
const char* double_excitation[8] = {"XXYX", "YXYY", "XYYY", "XXXY", "XYXX", "YYXY", "YYYX", "YXXX"};

qbit qubit_register[total_qubits];

quantum_kernel void uccsd() {
  for (int i = 0; i < total_qubits; i++) {
    PrepZ(qubit_register[i]);
  }
  /**/
  for (int p = 0; p < total_qubits; p++)
    for (int q = p+1; q < total_qubits; q++) {
        // {"XY", "YX"}
        // p
      {  // XY
        //p
        H(qubit_register[p]);
        // q
        {
          H(qubit_register[q]);
          Sdag(qubit_register[q]);
        }

        for (int i = p; i < q; i++)
          CNOT(qubit_register[i], qubit_register[i+1]);
        RZ(qubit_register[q], 1.0);
        for (int i = q-1; i >= p; i--)
          CNOT(qubit_register[i], qubit_register[i+1]);

        // p
        {
          H(qubit_register[p]);
        }
        // q
        {
          S(qubit_register[q]);
          H(qubit_register[q]);
        }
      }
      { // YX  
        {
          H(qubit_register[p]);
          Sdag(qubit_register[p]);
        }
        // // q
        {
          H(qubit_register[q]);
        }

        for (int i = p; i < q; i++)
          CNOT(qubit_register[i], qubit_register[i+1]);
        RZ(qubit_register[q], 1.0);
        for (int i = q-1; i >= p; i--)
          CNOT(qubit_register[i], qubit_register[i+1]);

        // p
        {
          S(qubit_register[p]);
          H(qubit_register[p]);
        }
        // q
        {
          H(qubit_register[q]);
        }
      }
    }
  /**/
  
  
  
  for (int p = 0; p < total_qubits; p++)
    for (int q = p+1; q < total_qubits; q++)
      for (int r = q+1; r < total_qubits; r++)
        for (int s = r+1; s < total_qubits; s++) {
          // {"XXYX", "YXYY", "XYYY", "XXXY", "XYXX", "YYXY", "YYYX", "YXXX"}
          { // XXYX
            // p
            {
              H(qubit_register[p]);
            }
            // q
            {
              H(qubit_register[q]);
            }
            // r
            {
              H(qubit_register[r]);
              Sdag(qubit_register[r]);
            }
            // s
            {
              H(qubit_register[s]);
            }
            
            for (int i = p; i < q; i++)
              CNOT(qubit_register[i], qubit_register[i+1]);
            CNOT(qubit_register[q], qubit_register[r]);
            for (int i = r; i < s; i++)
              CNOT(qubit_register[i], qubit_register[i+1]);
            RZ(qubit_register[s], 1.0);
            for (int i = s-1; i >= r; i--)
              CNOT(qubit_register[i], qubit_register[i+1]);
            CNOT(qubit_register[q], qubit_register[r]);
            for (int i = q-1; i >= p; i--)
              CNOT(qubit_register[i], qubit_register[i+1]);

            // p
            {
              H(qubit_register[p]);
            }
            // q
            {
              H(qubit_register[q]);
            }
            // r
            {
              S(qubit_register[r]);
              H(qubit_register[r]);
            }
            // s
            {
              H(qubit_register[s]);
            }
          }
          
          { // YXYY
            // p
            {
              H(qubit_register[p]);
              Sdag(qubit_register[p]);
            }
            // q
            {
              H(qubit_register[q]);
            }
            // r
            {
              H(qubit_register[r]);
              Sdag(qubit_register[r]);
            }
            // s
            {
              H(qubit_register[s]);
              Sdag(qubit_register[s]);
            }
            
            for (int i = p; i < q; i++)
              CNOT(qubit_register[i], qubit_register[i+1]);
            CNOT(qubit_register[q], qubit_register[r]);
            for (int i = r; i < s; i++)
              CNOT(qubit_register[i], qubit_register[i+1]);
            RZ(qubit_register[s], 1.0);
            for (int i = s-1; i >= r; i--)
              CNOT(qubit_register[i], qubit_register[i+1]);
            CNOT(qubit_register[q], qubit_register[r]);
            for (int i = q-1; i >= p; i--)
              CNOT(qubit_register[i], qubit_register[i+1]);

            // p
            {
              S(qubit_register[p]);
              H(qubit_register[p]);
            }
            // q
            {
              H(qubit_register[q]);
            }
            // r
            {
              S(qubit_register[r]);
              H(qubit_register[r]);
            }
            // s
            {
              S(qubit_register[s]);
              H(qubit_register[s]);
            }
          }
          
          { // XYYY
            // p
            {
              H(qubit_register[p]);
            }
            // q
            {
              H(qubit_register[q]);
              Sdag(qubit_register[q]);
            }
            // r
            {
              H(qubit_register[r]);
              Sdag(qubit_register[r]);
            }
            // s
            {
              H(qubit_register[s]);
              Sdag(qubit_register[s]);
            }
            
            for (int i = p; i < q; i++)
              CNOT(qubit_register[i], qubit_register[i+1]);
            CNOT(qubit_register[q], qubit_register[r]);
            for (int i = r; i < s; i++)
              CNOT(qubit_register[i], qubit_register[i+1]);
            RZ(qubit_register[s], 1.0);
            for (int i = s-1; i >= r; i--)
              CNOT(qubit_register[i], qubit_register[i+1]);
            CNOT(qubit_register[q], qubit_register[r]);
            for (int i = q-1; i >= p; i--)
              CNOT(qubit_register[i], qubit_register[i+1]);

            // p
            {
              H(qubit_register[p]);
            }
            // q
            {
              S(qubit_register[q]);
              H(qubit_register[q]);
            }
            // r
            {
              S(qubit_register[r]);
              H(qubit_register[r]);
            }
            // s
            {
              S(qubit_register[s]);
              H(qubit_register[s]);
            }
          }
          
          { // XXXY
            // p
            {
              H(qubit_register[p]);
            }
            // q
            {
              H(qubit_register[q]);
            }
            // r
            {
              H(qubit_register[r]);
            }
            // s
            {
              H(qubit_register[s]);
              Sdag(qubit_register[s]);
            }
            
            for (int i = p; i < q; i++)
              CNOT(qubit_register[i], qubit_register[i+1]);
            CNOT(qubit_register[q], qubit_register[r]);
            for (int i = r; i < s; i++)
              CNOT(qubit_register[i], qubit_register[i+1]);
            RZ(qubit_register[s], 1.0);
            for (int i = s-1; i >= r; i--)
              CNOT(qubit_register[i], qubit_register[i+1]);
            CNOT(qubit_register[q], qubit_register[r]);
            for (int i = q-1; i >= p; i--)
              CNOT(qubit_register[i], qubit_register[i+1]);

            // p
            {
              H(qubit_register[p]);
            }
            // q
            {
              H(qubit_register[q]);
            }
            // r
            {
              H(qubit_register[r]);
            }
            // s
            {
              S(qubit_register[s]);
              H(qubit_register[s]);
            }
          }
          
          { // XYXX
            // p
            {
              H(qubit_register[p]);
            }
            // q
            {
              H(qubit_register[q]);
              Sdag(qubit_register[q]);
            }
            // r
            {
              H(qubit_register[r]);
            }
            // s
            {
              H(qubit_register[s]);
            }
            
            for (int i = p; i < q; i++)
              CNOT(qubit_register[i], qubit_register[i+1]);
            CNOT(qubit_register[q], qubit_register[r]);
            for (int i = r; i < s; i++)
              CNOT(qubit_register[i], qubit_register[i+1]);
            RZ(qubit_register[s], 1.0);
            for (int i = s-1; i >= r; i--)
              CNOT(qubit_register[i], qubit_register[i+1]);
            CNOT(qubit_register[q], qubit_register[r]);
            for (int i = q-1; i >= p; i--)
              CNOT(qubit_register[i], qubit_register[i+1]);

            // p
            {
              H(qubit_register[p]);
            }
            // q
            {
              S(qubit_register[q]);
              H(qubit_register[q]);
            }
            // r
            {
              H(qubit_register[r]);
            }
            // s
            {
              H(qubit_register[s]);
            }
          }
          
          { // YYXY
            // p
            {
              H(qubit_register[p]);
              Sdag(qubit_register[p]);
            }
            // q
            {
              H(qubit_register[q]);
              Sdag(qubit_register[q]);
            }
            // r
            {
              H(qubit_register[r]);
            }
            // s
            {
              H(qubit_register[s]);
              Sdag(qubit_register[s]);
            }
            
            for (int i = p; i < q; i++)
              CNOT(qubit_register[i], qubit_register[i+1]);
            CNOT(qubit_register[q], qubit_register[r]);
            for (int i = r; i < s; i++)
              CNOT(qubit_register[i], qubit_register[i+1]);
            RZ(qubit_register[s], 1.0);
            for (int i = s-1; i >= r; i--)
              CNOT(qubit_register[i], qubit_register[i+1]);
            CNOT(qubit_register[q], qubit_register[r]);
            for (int i = q-1; i >= p; i--)
              CNOT(qubit_register[i], qubit_register[i+1]);

            // p
            {
              S(qubit_register[p]);
              H(qubit_register[p]);
            }
            // q
            {
              S(qubit_register[q]);
              H(qubit_register[q]);
            }
            // r
            {
              H(qubit_register[r]);
            }
            // s
            {
              S(qubit_register[s]);
              H(qubit_register[s]);
            }
          }
          
          { // YYYX
            // p
            {
              H(qubit_register[p]);
              Sdag(qubit_register[p]);
            }
            // q
            {
              H(qubit_register[q]);
              Sdag(qubit_register[q]);
            }
            // r
            {
              H(qubit_register[r]);
              Sdag(qubit_register[r]);
            }
            // s
            {
              H(qubit_register[s]);
            }
            
            for (int i = p; i < q; i++)
              CNOT(qubit_register[i], qubit_register[i+1]);
            CNOT(qubit_register[q], qubit_register[r]);
            for (int i = r; i < s; i++)
              CNOT(qubit_register[i], qubit_register[i+1]);
            RZ(qubit_register[s], 1.0);
            for (int i = s-1; i >= r; i--)
              CNOT(qubit_register[i], qubit_register[i+1]);
            CNOT(qubit_register[q], qubit_register[r]);
            for (int i = q-1; i >= p; i--)
              CNOT(qubit_register[i], qubit_register[i+1]);

            // p
            {
              S(qubit_register[p]);
              H(qubit_register[p]);
            }
            // q
            {
              S(qubit_register[q]);
              H(qubit_register[q]);
            }
            // r
            {
              S(qubit_register[r]);
              H(qubit_register[r]);
            }
            // s
            {
              H(qubit_register[s]);
            }
          }
          
          { // YXXX
            // p
            {
              H(qubit_register[p]);
              Sdag(qubit_register[p]);
            }
            // q
            {
              H(qubit_register[q]);
            }
            // r
            {
              H(qubit_register[r]);
            }
            // s
            {
              H(qubit_register[s]);
            }
            
            for (int i = p; i < q; i++)
              CNOT(qubit_register[i], qubit_register[i+1]);
            CNOT(qubit_register[q], qubit_register[r]);
            for (int i = r; i < s; i++)
              CNOT(qubit_register[i], qubit_register[i+1]);
            RZ(qubit_register[s], 1.0);
            for (int i = s-1; i >= r; i--)
              CNOT(qubit_register[i], qubit_register[i+1]);
            CNOT(qubit_register[q], qubit_register[r]);
            for (int i = q-1; i >= p; i--)
              CNOT(qubit_register[i], qubit_register[i+1]);

            // p
            {
              S(qubit_register[p]);
              H(qubit_register[p]);
            }
            // q
            {
              H(qubit_register[q]);
            }
            // r
            {
              H(qubit_register[r]);
            }
            // s
            {
              H(qubit_register[s]);
            }
          }
        }

}

int main() {
  iqsdk::IqsConfig settings(total_qubits, "noiseless");
  settings.verbose = true;
  iqsdk::FullStateSimulator quantum_8086(settings);
  if (iqsdk::QRT_ERROR_SUCCESS != quantum_8086.ready())
    return 1;

  // get references to qbits
  std::vector<std::reference_wrapper<qbit>> qids;
  for (int id = 0; id < total_qubits; ++id) {
    qids.push_back(std::ref(qubit_register[id]));
  }

  uccsd();

}
