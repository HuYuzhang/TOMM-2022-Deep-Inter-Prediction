/* The copyright in this software is being made available under the BSD
 * License, included below. This software may be subject to other third party
 * and contributor rights, including patent rights, and no such rights are
 * granted under this license.
 *
 * Copyright (c) 2010-2021, ITU/ISO/IEC
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 *  * Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *  * Neither the name of the ITU/ISO/IEC nor the names of its contributors may
 *    be used to endorse or promote products derived from this software without
 *    specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS
 * BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF
 * THE POSSIBILITY OF SUCH DAMAGE.
 */

/** \file     SEIFilmGrainAnalyzer.h
    \brief    SMPTE RDD5 based film grain analysis functionality from SEI messages
*/

#ifndef __SEIFILMGRAINANALYZER__
#define __SEIFILMGRAINANALYZER__

#pragma once

#include "CommonLib/Picture.h"
#include "CommonLib/SEI.h"
#include "Utilities/VideoIOYuv.h"
#include "CommonLib/CommonDef.h"

#include "TrQuant_EMT.h"

#include <numeric>
#include <cmath>
#include <algorithm>

#if JVET_X0048_X0103_FILM_GRAIN

static const int      MAX_REAL_SCALE                =     32;
static const double   PI                            =     3.14159265358979323846;

// POLYFIT
static const int      MAXPAIRS                                  = 256;
static const int      MAXORDER                                  = 8;     // maximum order of polinomial fitting
static const int      ORDER                                     = 3;     // order of polinomial function
static const int      QUANT_LEVELS                              = 4;     // number of quantization levels in lloyd max quantization
static const int      INTERVAL_SIZE                             = 16;
static const int      MIN_ELEMENT_NUMBER_PER_INTENSITY_INTERVAL = 32;
static const int      MIN_POINTS_FOR_INTENSITY_ESTIMATION       = 128;   // 4*32 = 128; 4 intervals with at least 32 points
static const int      MIN_BLOCKS_FOR_CUTOFF_ESTIMATION          = 4;     // 4 blocks of 64 x 64 size
static const int      POINT_STEP                                = 8;     // step size in point extension
static const double   POINT_SCALE                               = 1.5;   // scaling in point extension
static const double   VAR_SCALE_DOWN                            = 1.5;   // filter out large points
static const double   VAR_SCALE_UP                              = 0.5;   // filter out large points

//! \ingroup SEIFilmGrainAnalyzer
//! \{

// ====================================================================================================================
// Class definition
// ====================================================================================================================

struct Picture;

typedef std::vector<std::vector<Intermediate_Int>> PelMatrix;
typedef std::vector<std::vector<double>>           PelMatrixDouble;

typedef std::vector<std::vector<long double>>      PelMatrixLongDouble;
typedef std::vector<long double>                   PelVectorLongDouble;

class Canny
{
public:
  Canny();
  ~Canny();

  unsigned int      m_convWidthG = 5, m_convHeightG = 5;		  // Pixel's row and col positions for Gauss filtering

  void detect_edges(const PelStorage* orig, PelStorage* dest, unsigned int uiBitDepth, ComponentID compID);

private:
  static const int  m_gx[3][3];                               // Sobel kernel x
  static const int  m_gy[3][3];                               // Sobel kernel y
  static const int  m_gauss5x5[5][5];                         // Gauss 5x5 kernel, integer approximation

  unsigned int      m_convWidthS = 3, m_convHeightS = 3;		  // Pixel's row and col positions for Sobel filtering
  double            m_sigma      = 1.0;                       // Gaussian filter sigma

  double            m_lowThresholdRatio   = 0.1;               // low threshold rato
  int               m_highThresholdRatio  = 3;                 // high threshold rato
  
  void gradient   ( PelStorage* buff1, PelStorage* buff2,
                    unsigned int width, unsigned int height,
                    unsigned int convWidthS, unsigned int convHeightS, unsigned int bitDepth, ComponentID compID );
  void suppressNonMax ( PelStorage* buff1, PelStorage* buff2, unsigned int width, unsigned int height, ComponentID compID );
  void doubleThreshold( PelStorage *buff, unsigned int width, unsigned int height, /*unsigned int windowSizeRatio,*/
                       unsigned int bitDepth, ComponentID compID);
  void edgeTracking   ( PelStorage* buff1, unsigned int width, unsigned int height,
                       unsigned int windowWidth, unsigned int windowHeight, unsigned int bitDepth, ComponentID compID );
};


class Morph
{
public:
  Morph();
  ~Morph();

  int dilation  (PelStorage* buff, unsigned int bitDepth, ComponentID compID, int numIter, int iter = 0);
  int erosion   (PelStorage* buff, unsigned int bitDepth, ComponentID compID, int numIter, int iter = 0);

private:
  unsigned int m_kernelSize = 3;		// Dilation and erosion kernel size
};


class FGAnalyser
{
public:
  FGAnalyser();
  ~FGAnalyser();

  void init(const int width,
            const int height,
            const ChromaFormat inputChroma,
            const BitDepths& inputBitDepths,
            const bool doAnalysis[]);
  void destroy        ();
  void initBufs       (Picture* pic);
  void estimate_grain (Picture* pic);

  int                                     getLog2scaleFactor()  { return m_log2ScaleFactor; };
  SEIFilmGrainCharacteristics::CompModel  getCompModel(int idx) { return m_compModel[idx];  };

private:
  ChromaFormat  m_chromaFormatIDC;
  BitDepths     m_bitDepths;
  bool          m_doAnalysis[MAX_NUM_COMPONENT] = { false, false, false };

  Canny    m_edgeDetector;
  Morph    m_morphOperation;
  double   m_lowIntensityRatio            = 0.1;                    // supress everything below 0.1*maxIntensityOffset
  static constexpr double m_tap_filtar[3] = { 1, 2, 1 };
  static constexpr double m_normTap       = 4.0;

  // fg model parameters
  int                                    m_log2ScaleFactor;
  SEIFilmGrainCharacteristics::CompModel m_compModel[MAX_NUM_COMPONENT];

  PelStorage *m_originalBuf = nullptr;
  PelStorage *m_workingBuf  = nullptr;
  PelStorage *m_maskBuf     = nullptr;

  int  denoise                  (Picture* pic);
  void findMask                 ();

  void estimate_grain_parameters    ();
  void block_transform              (const PelStorage& buff1, std::vector<PelMatrix>& squared_dct_grain_block_list, int offsetX, int offsetY, unsigned int bitDepth, ComponentID compID);
  void estimate_cutoff_freq         (const std::vector<PelMatrix>& blocks, ComponentID compID);
  int  cutoff_frequency             (std::vector<double>& mean);
  void estimate_scaling_factors     (std::vector<int>& data_x, std::vector<int>& data_y, unsigned int bitDepth, ComponentID compID);
  bool fit_function                 (std::vector<int>& data_x, std::vector<int>& data_y, std::vector<double>& coeffs, std::vector<double>& scalingVec,
                                     int order, int bitDepth, bool second_pass);
  void avg_scaling_vec              (std::vector<double> &scalingVec, ComponentID compID, int bitDepth);
  bool lloyd_max                    (std::vector<double>& scalingVec, std::vector<int>& quantizedVec, double& distortion, int numQuantizedLevels, int bitDepth);
  void quantize                     (std::vector<double>& scalingVec, std::vector<double>& quantizedVec, double& distortion, std::vector<double> partition, std::vector<double> codebook);
  void extend_points                (std::vector<int>& data_x, std::vector<int>& data_y, int bitDepth);

  void setEstimatedParameters       (std::vector<int>& quantizedVec, unsigned int bitDepth, ComponentID compID);
  void define_intervals_and_scalings(std::vector<std::vector<int>>& parameters, std::vector<int>& quantizedVec, int bitDepth);
  void scale_down                   (std::vector<std::vector<int>>& parameters, int bitDepth);
  void confirm_intervals            (std::vector<std::vector<int>>& parameters);

  long double ldpow                 (long double n, unsigned p);
  int         meanVar               (PelStorage& buffer, int windowSize, ComponentID compID, int offsetX, int offsetY, bool getVar);
  int         count_edges           (PelStorage& buffer, int windowSize, ComponentID compID, int offsetX, int offsetY);

  void subsample                    (const PelStorage& input, PelStorage& output, ComponentID compID, const int factor = 2, const int padding = 0) const;
  void upsample                     (const PelStorage& input, PelStorage& output, ComponentID compID, const int factor = 2, const int padding = 0) const;
  void combineMasks                 (PelStorage& buff, PelStorage& buff2, ComponentID compID);
  void suppressLowIntensity         (const PelStorage& buff1, PelStorage& buff2, unsigned int bitDepth, ComponentID compID);

}; // END CLASS DEFINITION

//! \}
#endif

#endif // __SEIFILMGRAINANALYZER__


