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

#ifdef ENABLE_VTK_OUTPUT

#include "vtk_output.h"
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <string>
#include <vector>

// Global variables from main.c
extern char path[256];
extern int levelset;
extern PetscInt ti;

// VTK output options
PetscInt vtk_output = 0;
PetscInt vtk_binary = 0;  // Default to ASCII for readability

extern "C" int VTK_Initialize(void) {
    PetscOptionsGetInt(NULL, NULL, "-vtk_output", &vtk_output, NULL);
    PetscOptionsGetInt(NULL, NULL, "-vtk_binary", &vtk_binary, NULL);
    return 0;
}

extern "C" int VTK_Finalize(void) {
    return 0;
}

/**
 * Write a combined VTK file gathering all data to rank 0.
 * Uses DMDACreateNaturalVector and proper ordering conversion.
 *
 * Output format is human-readable ASCII VTK Legacy format with:
 * - Coordinates (X, Y, Z)
 * - Velocity vector (U, V, W)
 * - Pressure scalar
 * - Nvert (blanking) scalar
 * - Levelset scalar (if enabled)
 *
 * Output: output_XXXXXX.vtk (single file)
 */
extern "C" int VTK_WriteStructuredGridCombined(UserCtx* user, int timestep) {
    int rank, nprocs;
    MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
    MPI_Comm_size(PETSC_COMM_WORLD, &nprocs);

    DMDALocalInfo info = user->info;

    // Global dimensions
    PetscInt mx = info.mx, my = info.my, mz = info.mz;
    PetscInt num_points = mx * my * mz;

    // Get DMs
    DM da = user->da;
    DM fda = user->fda;

    // Get global coordinate vector
    Vec gCoor;
    DMGetCoordinates(da, &gCoor);

    // Create global vectors for fields
    Vec gUcat, gP, gNvert, gLevelset = NULL;
    DMCreateGlobalVector(fda, &gUcat);
    DMCreateGlobalVector(da, &gP);
    DMCreateGlobalVector(da, &gNvert);

    // Copy local to global
    DMLocalToGlobal(fda, user->lUcat, INSERT_VALUES, gUcat);
    DMLocalToGlobal(da, user->lP, INSERT_VALUES, gP);
    DMLocalToGlobal(da, user->lNvert, INSERT_VALUES, gNvert);

    if (levelset) {
        DMCreateGlobalVector(da, &gLevelset);
        DMLocalToGlobal(da, user->lLevelset, INSERT_VALUES, gLevelset);
    }

    // Create natural vectors (proper i-j-k ordering)
    Vec natCoor, natUcat, natP, natNvert, natLevelset = NULL;
    DMDACreateNaturalVector(fda, &natCoor);
    DMDACreateNaturalVector(fda, &natUcat);
    DMDACreateNaturalVector(da, &natP);
    DMDACreateNaturalVector(da, &natNvert);

    // Convert from global (parallel) to natural (i-j-k) ordering
    DMDAGlobalToNaturalBegin(fda, gCoor, INSERT_VALUES, natCoor);
    DMDAGlobalToNaturalEnd(fda, gCoor, INSERT_VALUES, natCoor);

    DMDAGlobalToNaturalBegin(fda, gUcat, INSERT_VALUES, natUcat);
    DMDAGlobalToNaturalEnd(fda, gUcat, INSERT_VALUES, natUcat);

    DMDAGlobalToNaturalBegin(da, gP, INSERT_VALUES, natP);
    DMDAGlobalToNaturalEnd(da, gP, INSERT_VALUES, natP);

    DMDAGlobalToNaturalBegin(da, gNvert, INSERT_VALUES, natNvert);
    DMDAGlobalToNaturalEnd(da, gNvert, INSERT_VALUES, natNvert);

    if (levelset) {
        DMDACreateNaturalVector(da, &natLevelset);
        DMDAGlobalToNaturalBegin(da, gLevelset, INSERT_VALUES, natLevelset);
        DMDAGlobalToNaturalEnd(da, gLevelset, INSERT_VALUES, natLevelset);
    }

    // Now scatter natural vectors to rank 0
    Vec seqCoor, seqUcat, seqP, seqNvert, seqLevelset = NULL;
    VecScatter scatterCoor, scatterUcat, scatterP, scatterNvert, scatterLevelset;

    VecScatterCreateToZero(natCoor, &scatterCoor, &seqCoor);
    VecScatterCreateToZero(natUcat, &scatterUcat, &seqUcat);
    VecScatterCreateToZero(natP, &scatterP, &seqP);
    VecScatterCreateToZero(natNvert, &scatterNvert, &seqNvert);

    VecScatterBegin(scatterCoor, natCoor, seqCoor, INSERT_VALUES, SCATTER_FORWARD);
    VecScatterBegin(scatterUcat, natUcat, seqUcat, INSERT_VALUES, SCATTER_FORWARD);
    VecScatterBegin(scatterP, natP, seqP, INSERT_VALUES, SCATTER_FORWARD);
    VecScatterBegin(scatterNvert, natNvert, seqNvert, INSERT_VALUES, SCATTER_FORWARD);

    VecScatterEnd(scatterCoor, natCoor, seqCoor, INSERT_VALUES, SCATTER_FORWARD);
    VecScatterEnd(scatterUcat, natUcat, seqUcat, INSERT_VALUES, SCATTER_FORWARD);
    VecScatterEnd(scatterP, natP, seqP, INSERT_VALUES, SCATTER_FORWARD);
    VecScatterEnd(scatterNvert, natNvert, seqNvert, INSERT_VALUES, SCATTER_FORWARD);

    if (levelset) {
        VecScatterCreateToZero(natLevelset, &scatterLevelset, &seqLevelset);
        VecScatterBegin(scatterLevelset, natLevelset, seqLevelset, INSERT_VALUES, SCATTER_FORWARD);
        VecScatterEnd(scatterLevelset, natLevelset, seqLevelset, INSERT_VALUES, SCATTER_FORWARD);
    }

    // Only rank 0 writes the file
    if (rank == 0) {
        PetscScalar *coorArr, *ucatArr, *pArr, *nvertArr, *levelArr = NULL;
        VecGetArray(seqCoor, &coorArr);
        VecGetArray(seqUcat, &ucatArr);
        VecGetArray(seqP, &pArr);
        VecGetArray(seqNvert, &nvertArr);
        if (levelset && seqLevelset) {
            VecGetArray(seqLevelset, &levelArr);
        }

        char filename[256];
        snprintf(filename, sizeof(filename), "%s/output_%06d.vtk", path, timestep);

        FILE* fp = fopen(filename, "w");
        if (!fp) {
            PetscPrintf(PETSC_COMM_SELF, "Error: Cannot open file %s for writing\n", filename);
            VecRestoreArray(seqCoor, &coorArr);
            VecRestoreArray(seqUcat, &ucatArr);
            VecRestoreArray(seqP, &pArr);
            VecRestoreArray(seqNvert, &nvertArr);
            if (levelset && levelArr) VecRestoreArray(seqLevelset, &levelArr);
            goto cleanup;
        }

        // Write VTK Legacy header
        fprintf(fp, "# vtk DataFile Version 3.0\n");
        fprintf(fp, "VFS-Wind output timestep %d\n", timestep);
        fprintf(fp, "ASCII\n");
        fprintf(fp, "DATASET STRUCTURED_GRID\n");
        fprintf(fp, "DIMENSIONS %d %d %d\n", (int)mx, (int)my, (int)mz);

        // Write POINTS (coordinates)
        // Natural ordering: i varies fastest, then j, then k
        fprintf(fp, "POINTS %d double\n", (int)num_points);
        for (PetscInt idx = 0; idx < num_points; idx++) {
            fprintf(fp, "%.10e %.10e %.10e\n",
                    (double)coorArr[3*idx + 0],
                    (double)coorArr[3*idx + 1],
                    (double)coorArr[3*idx + 2]);
        }

        // Write POINT_DATA
        fprintf(fp, "\nPOINT_DATA %d\n", (int)num_points);

        // Velocity as VECTORS
        fprintf(fp, "VECTORS Velocity double\n");
        for (PetscInt idx = 0; idx < num_points; idx++) {
            fprintf(fp, "%.10e %.10e %.10e\n",
                    (double)ucatArr[3*idx + 0],
                    (double)ucatArr[3*idx + 1],
                    (double)ucatArr[3*idx + 2]);
        }

        // Pressure as SCALARS
        fprintf(fp, "SCALARS Pressure double 1\n");
        fprintf(fp, "LOOKUP_TABLE default\n");
        for (PetscInt idx = 0; idx < num_points; idx++) {
            fprintf(fp, "%.10e\n", (double)pArr[idx]);
        }

        // Nvert as SCALARS
        fprintf(fp, "SCALARS Nvert double 1\n");
        fprintf(fp, "LOOKUP_TABLE default\n");
        for (PetscInt idx = 0; idx < num_points; idx++) {
            fprintf(fp, "%.10e\n", (double)nvertArr[idx]);
        }

        // Levelset as SCALARS (if enabled)
        if (levelset && levelArr) {
            fprintf(fp, "SCALARS Levelset double 1\n");
            fprintf(fp, "LOOKUP_TABLE default\n");
            for (PetscInt idx = 0; idx < num_points; idx++) {
                fprintf(fp, "%.10e\n", (double)levelArr[idx]);
            }
        }

        fclose(fp);

        VecRestoreArray(seqCoor, &coorArr);
        VecRestoreArray(seqUcat, &ucatArr);
        VecRestoreArray(seqP, &pArr);
        VecRestoreArray(seqNvert, &nvertArr);
        if (levelset && levelArr) {
            VecRestoreArray(seqLevelset, &levelArr);
        }

        PetscPrintf(PETSC_COMM_WORLD, "VTK output written: %s\n", filename);
    }

cleanup:
    // Cleanup scatter contexts and sequential vectors
    VecScatterDestroy(&scatterCoor);
    VecScatterDestroy(&scatterUcat);
    VecScatterDestroy(&scatterP);
    VecScatterDestroy(&scatterNvert);
    if (seqCoor) VecDestroy(&seqCoor);
    if (seqUcat) VecDestroy(&seqUcat);
    if (seqP) VecDestroy(&seqP);
    if (seqNvert) VecDestroy(&seqNvert);

    // Cleanup natural vectors
    VecDestroy(&natCoor);
    VecDestroy(&natUcat);
    VecDestroy(&natP);
    VecDestroy(&natNvert);

    if (levelset) {
        VecScatterDestroy(&scatterLevelset);
        if (seqLevelset) VecDestroy(&seqLevelset);
        VecDestroy(&natLevelset);
        VecDestroy(&gLevelset);
    }

    // Cleanup global vectors we created
    VecDestroy(&gUcat);
    VecDestroy(&gP);
    VecDestroy(&gNvert);

    MPI_Barrier(PETSC_COMM_WORLD);

    return 0;
}

/**
 * Write VTK Legacy ASCII Structured Grid file (per-process version).
 * Each process writes its own local portion directly using DMDAVecGetArray.
 * This preserves correct ordering within each piece.
 */
extern "C" int VTK_WriteStructuredGrid(UserCtx* user, int timestep) {
    int rank, nprocs;
    MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
    MPI_Comm_size(PETSC_COMM_WORLD, &nprocs);

    DMDALocalInfo info = user->info;

    // Get local domain dimensions (without ghost points)
    PetscInt xs = info.xs, xm = info.xm;
    PetscInt ys = info.ys, ym = info.ym;
    PetscInt zs = info.zs, zm = info.zm;

    // Get coordinates
    Vec Coor;
    DMGetCoordinatesLocal(user->da, &Coor);
    Cmpnts*** coor;
    DMDAVecGetArray(user->fda, Coor, &coor);

    // Get velocity
    Cmpnts*** ucat;
    DMDAVecGetArray(user->fda, user->lUcat, &ucat);

    // Get pressure
    PetscReal*** p;
    DMDAVecGetArray(user->da, user->lP, &p);

    // Get nvert
    PetscReal*** nvert;
    DMDAVecGetArray(user->da, user->lNvert, &nvert);

    // Get levelset if enabled
    PetscReal*** level = NULL;
    if (levelset) {
        DMDAVecGetArray(user->da, user->lLevelset, &level);
    }

    // Calculate number of points
    PetscInt num_points = xm * ym * zm;

    // Open output file
    char filename[256];
    snprintf(filename, sizeof(filename), "%s/output_%06d_p%d.vtk", path, timestep, rank);

    FILE* fp = fopen(filename, "w");
    if (!fp) {
        PetscPrintf(PETSC_COMM_SELF, "Error: Cannot open file %s for writing\n", filename);
        return -1;
    }

    // Write VTK Legacy header
    fprintf(fp, "# vtk DataFile Version 3.0\n");
    fprintf(fp, "VFS-Wind output timestep %d rank %d\n", timestep, rank);
    fprintf(fp, "ASCII\n");
    fprintf(fp, "DATASET STRUCTURED_GRID\n");
    fprintf(fp, "DIMENSIONS %d %d %d\n", (int)xm, (int)ym, (int)zm);

    // Write POINTS (coordinates) - i varies fastest, then j, then k
    fprintf(fp, "POINTS %d double\n", (int)num_points);
    for (PetscInt k = zs; k < zs + zm; k++) {
        for (PetscInt j = ys; j < ys + ym; j++) {
            for (PetscInt i = xs; i < xs + xm; i++) {
                fprintf(fp, "%.10e %.10e %.10e\n",
                        coor[k][j][i].x, coor[k][j][i].y, coor[k][j][i].z);
            }
        }
    }

    // Write POINT_DATA section
    fprintf(fp, "\nPOINT_DATA %d\n", (int)num_points);

    // Write Velocity as VECTORS
    fprintf(fp, "VECTORS Velocity double\n");
    for (PetscInt k = zs; k < zs + zm; k++) {
        for (PetscInt j = ys; j < ys + ym; j++) {
            for (PetscInt i = xs; i < xs + xm; i++) {
                fprintf(fp, "%.10e %.10e %.10e\n",
                        ucat[k][j][i].x, ucat[k][j][i].y, ucat[k][j][i].z);
            }
        }
    }

    // Write Pressure as SCALARS
    fprintf(fp, "SCALARS Pressure double 1\n");
    fprintf(fp, "LOOKUP_TABLE default\n");
    for (PetscInt k = zs; k < zs + zm; k++) {
        for (PetscInt j = ys; j < ys + ym; j++) {
            for (PetscInt i = xs; i < xs + xm; i++) {
                fprintf(fp, "%.10e\n", p[k][j][i]);
            }
        }
    }

    // Write Nvert as SCALARS
    fprintf(fp, "SCALARS Nvert double 1\n");
    fprintf(fp, "LOOKUP_TABLE default\n");
    for (PetscInt k = zs; k < zs + zm; k++) {
        for (PetscInt j = ys; j < ys + ym; j++) {
            for (PetscInt i = xs; i < xs + xm; i++) {
                fprintf(fp, "%.10e\n", nvert[k][j][i]);
            }
        }
    }

    // Write Levelset as SCALARS (if enabled)
    if (levelset && level) {
        fprintf(fp, "SCALARS Levelset double 1\n");
        fprintf(fp, "LOOKUP_TABLE default\n");
        for (PetscInt k = zs; k < zs + zm; k++) {
            for (PetscInt j = ys; j < ys + ym; j++) {
                for (PetscInt i = xs; i < xs + xm; i++) {
                    fprintf(fp, "%.10e\n", level[k][j][i]);
                }
            }
        }
    }

    fclose(fp);

    // Restore arrays
    DMDAVecRestoreArray(user->fda, Coor, &coor);
    DMDAVecRestoreArray(user->fda, user->lUcat, &ucat);
    DMDAVecRestoreArray(user->da, user->lP, &p);
    DMDAVecRestoreArray(user->da, user->lNvert, &nvert);
    if (levelset && level) {
        DMDAVecRestoreArray(user->da, user->lLevelset, &level);
    }

    // Write index file from rank 0
    MPI_Barrier(PETSC_COMM_WORLD);

    if (rank == 0) {
        char index_filename[256];
        snprintf(index_filename, sizeof(index_filename), "%s/output_%06d.vtk.index", path, timestep);
        FILE* idx = fopen(index_filename, "w");
        if (idx) {
            fprintf(idx, "# VFS-Wind VTK Output Index\n");
            fprintf(idx, "# Timestep: %d\n", timestep);
            fprintf(idx, "# Number of processes: %d\n", nprocs);
            fprintf(idx, "# Files:\n");
            for (int p = 0; p < nprocs; p++) {
                fprintf(idx, "output_%06d_p%d.vtk\n", timestep, p);
            }
            fclose(idx);
        }

        PetscPrintf(PETSC_COMM_WORLD, "VTK output written: output_%06d_p*.vtk (%d files)\n", timestep, nprocs);
    }

    return 0;
}

#endif /* ENABLE_VTK_OUTPUT */
