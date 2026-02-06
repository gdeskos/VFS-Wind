/*****************************************************************
* Copyright (C) by Regents of the University of Minnesota.       *
*                                                                *
* This Software is released under GNU General Public License 2.0 *
* http://www.gnu.org/licenses/gpl-2.0.html                       *
*                                                                *
* Modernization of the code by G Deskos,                         *
* Parametrica Research & Analytics                               *
*                                                                *
******************************************************************/

#ifndef _VTK_OUTPUT_H_
#define _VTK_OUTPUT_H_

#ifdef ENABLE_VTK_OUTPUT

#include "variables.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Write structured grid data in VTK Legacy ASCII format (.vtk).
 *
 * Output format is human-readable ASCII with:
 * - Coordinates (X, Y, Z)
 * - Velocity vector (U, V, W)
 * - Pressure scalar
 * - Nvert (blanking) scalar
 * - Levelset scalar (if enabled)
 *
 * For parallel runs, creates one file per MPI process:
 * - output_XXXXXX_p0.vtk, output_XXXXXX_p1.vtk, etc.
 * - output_XXXXXX.vtk.index (index file listing all pieces)
 *
 * @param user Pointer to UserCtx structure
 * @param timestep Current simulation timestep
 * @return 0 on success, non-zero on failure
 */
int VTK_WriteStructuredGrid(UserCtx* user, int timestep);

/**
 * Write a combined VTK file gathering all data to rank 0.
 *
 * Creates a single output file containing the complete domain.
 * Use this for smaller problems or when a single file is preferred.
 *
 * Output: output_XXXXXX.vtk
 *
 * @param user Pointer to UserCtx structure
 * @param timestep Current simulation timestep
 * @return 0 on success, non-zero on failure
 */
int VTK_WriteStructuredGridCombined(UserCtx* user, int timestep);

/**
 * Initialize VTK output system.
 * Reads command-line options:
 *   -vtk_output <0|1>  Enable/disable VTK output (default: 0)
 *   -vtk_binary <0|1>  Use binary format (default: 0, ASCII)
 *
 * @return 0 on success, non-zero on failure
 */
int VTK_Initialize(void);

/**
 * Finalize VTK output system.
 * @return 0 on success, non-zero on failure
 */
int VTK_Finalize(void);

#ifdef __cplusplus
}
#endif

#endif /* ENABLE_VTK_OUTPUT */

#endif /* _VTK_OUTPUT_H_ */
