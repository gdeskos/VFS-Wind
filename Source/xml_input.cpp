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

#ifdef ENABLE_XML_INPUT

#include "xml_input.h"
#include "tinyxml2/tinyxml2.h"
#include <cstdio>
#include <cstring>
#include <string>
#include <sys/stat.h>
#include "petscsys.h"

using namespace tinyxml2;

// Helper function to check if file exists
extern "C" int xml_file_exists(const char* filename) {
    struct stat buffer;
    return (stat(filename, &buffer) == 0) ? 1 : 0;
}

// Helper function to set a PETSc option
static void SetPetscOption(const char* name, const char* value) {
    if (value && strlen(value) > 0) {
        PetscOptionsSetValue(NULL, name, value);
    }
}

static void SetPetscOptionInt(const char* name, int value) {
    char buf[64];
    snprintf(buf, sizeof(buf), "%d", value);
    PetscOptionsSetValue(NULL, name, buf);
}

static void SetPetscOptionReal(const char* name, double value) {
    char buf[64];
    snprintf(buf, sizeof(buf), "%.15e", value);
    PetscOptionsSetValue(NULL, name, buf);
}

// Parse simulation section
static void ParseSimulation(XMLElement* root) {
    XMLElement* sim = root->FirstChildElement("simulation");
    if (!sim) return;

    XMLElement* timestep = sim->FirstChildElement("timestep");
    if (timestep) {
        double dt = 0.0;
        int totalsteps = 0;
        int output_interval = 10;

        if (timestep->QueryDoubleAttribute("dt", &dt) == XML_SUCCESS) {
            SetPetscOptionReal("-dt", dt);
        }
        if (timestep->QueryIntAttribute("totalsteps", &totalsteps) == XML_SUCCESS) {
            SetPetscOptionInt("-ti", totalsteps);
        }
        if (timestep->QueryIntAttribute("output_interval", &output_interval) == XML_SUCCESS) {
            SetPetscOptionInt("-tio", output_interval);
        }
    }

    XMLElement* restart = sim->FirstChildElement("restart");
    if (restart) {
        int enabled = 0;
        int timestep_val = 0;
        if (restart->QueryIntAttribute("enabled", &enabled) == XML_SUCCESS && enabled) {
            if (restart->QueryIntAttribute("timestep", &timestep_val) == XML_SUCCESS) {
                SetPetscOptionInt("-rstart", timestep_val);
            }
        }
    }
}

// Parse physics section
static void ParsePhysics(XMLElement* root) {
    XMLElement* physics = root->FirstChildElement("physics");
    if (!physics) return;

    XMLElement* reynolds = physics->FirstChildElement("reynolds_number");
    if (reynolds) {
        double re = 0.0;
        if (reynolds->QueryDoubleText(&re) == XML_SUCCESS) {
            SetPetscOptionReal("-ren", re);
        }
    }

    XMLElement* gravity = physics->FirstChildElement("gravity");
    if (gravity) {
        double gx = 0.0, gy = 0.0, gz = 0.0;
        gravity->QueryDoubleAttribute("x", &gx);
        gravity->QueryDoubleAttribute("y", &gy);
        gravity->QueryDoubleAttribute("z", &gz);
        SetPetscOptionReal("-gravity_x", gx);
        SetPetscOptionReal("-gravity_y", gy);
        SetPetscOptionReal("-gravity_z", gz);
    }
}

// Parse turbulence section
static void ParseTurbulence(XMLElement* root) {
    XMLElement* turb = root->FirstChildElement("turbulence");
    if (!turb) return;

    XMLElement* les = turb->FirstChildElement("les");
    if (les) {
        int enabled = 0;
        if (les->QueryIntAttribute("enabled", &enabled) == XML_SUCCESS && enabled) {
            SetPetscOptionInt("-les", enabled);
            const char* model = les->Attribute("model");
            if (model) {
                if (strcmp(model, "smagorinsky") == 0) {
                    SetPetscOptionInt("-les", 1);
                } else if (strcmp(model, "dynamic") == 0) {
                    SetPetscOptionInt("-les", 2);
                }
            }
            double cs = 0.1;
            if (les->QueryDoubleAttribute("cs", &cs) == XML_SUCCESS) {
                SetPetscOptionReal("-cs", cs);
            }
        }
    }

    XMLElement* rans = turb->FirstChildElement("rans");
    if (rans) {
        int enabled = 0;
        if (rans->QueryIntAttribute("enabled", &enabled) == XML_SUCCESS && enabled) {
            SetPetscOptionInt("-rans", 1);
        }
    }

    XMLElement* wf = turb->FirstChildElement("wall_function");
    if (wf) {
        int enabled = 0;
        if (wf->QueryIntAttribute("enabled", &enabled) == XML_SUCCESS && enabled) {
            SetPetscOptionInt("-wallfunction", 1);
        }
    }
}

// Parse levelset section
static void ParseLevelset(XMLElement* root) {
    XMLElement* ls = root->FirstChildElement("levelset");
    if (!ls) return;

    int enabled = 0;
    if (ls->QueryIntAttribute("enabled", &enabled) == XML_SUCCESS && enabled) {
        SetPetscOptionInt("-levelset", 1);
    }

    XMLElement* fluid0 = ls->FirstChildElement("fluid0");
    if (fluid0) {
        double density = 0.0, viscosity = 0.0;
        if (fluid0->QueryDoubleAttribute("density", &density) == XML_SUCCESS) {
            SetPetscOptionReal("-rho_water", density);
        }
        if (fluid0->QueryDoubleAttribute("viscosity", &viscosity) == XML_SUCCESS) {
            SetPetscOptionReal("-mu_water", viscosity);
        }
    }

    XMLElement* fluid1 = ls->FirstChildElement("fluid1");
    if (fluid1) {
        double density = 0.0, viscosity = 0.0;
        if (fluid1->QueryDoubleAttribute("density", &density) == XML_SUCCESS) {
            SetPetscOptionReal("-rho_air", density);
        }
        if (fluid1->QueryDoubleAttribute("viscosity", &viscosity) == XML_SUCCESS) {
            SetPetscOptionReal("-mu_air", viscosity);
        }
    }

    XMLElement* iterations = ls->FirstChildElement("iterations");
    if (iterations) {
        int it = 10;
        if (iterations->QueryIntText(&it) == XML_SUCCESS) {
            SetPetscOptionInt("-levelset_it", it);
        }
    }
}

// Parse solvers section
static void ParseSolvers(XMLElement* root) {
    XMLElement* solvers = root->FirstChildElement("solvers");
    if (!solvers) return;

    XMLElement* poisson = solvers->FirstChildElement("poisson");
    if (poisson) {
        int type = 1;
        int iterations = 15;
        double tolerance = 5e-9;

        poisson->QueryIntAttribute("type", &type);
        poisson->QueryIntAttribute("iterations", &iterations);
        poisson->QueryDoubleAttribute("tolerance", &tolerance);

        SetPetscOptionInt("-poisson", type);
        SetPetscOptionInt("-poisson_it", iterations);
        SetPetscOptionReal("-poisson_threshold", tolerance);
    }

    XMLElement* momentum = solvers->FirstChildElement("momentum");
    if (momentum) {
        int implicit = 0;
        int max_iterations = 50;

        momentum->QueryIntAttribute("implicit", &implicit);
        momentum->QueryIntAttribute("max_iterations", &max_iterations);

        if (implicit > 0) {
            SetPetscOptionInt("-imp", implicit);
            SetPetscOptionInt("-imp_MAX_IT", max_iterations);
        }
    }
}

// Parse parallel section
static void ParseParallel(XMLElement* root) {
    XMLElement* parallel = root->FirstChildElement("parallel");
    if (!parallel) return;

    XMLElement* periodic = parallel->FirstChildElement("periodic");
    if (periodic) {
        int i = 0, j = 0, k = 0;
        periodic->QueryIntAttribute("i", &i);
        periodic->QueryIntAttribute("j", &j);
        periodic->QueryIntAttribute("k", &k);

        if (i) SetPetscOptionInt("-i_periodic", i);
        if (j) SetPetscOptionInt("-j_periodic", j);
        if (k) SetPetscOptionInt("-k_periodic", k);
    }
}

// Parse output section
static void ParseOutput(XMLElement* root) {
    XMLElement* output = root->FirstChildElement("output");
    if (!output) return;

    XMLElement* path = output->FirstChildElement("path");
    if (path) {
        const char* pathText = path->GetText();
        if (pathText) {
            SetPetscOption("-path", pathText);
        }
    }

    XMLElement* vtk = output->FirstChildElement("vtk");
    if (vtk) {
        int enabled = 0;
        if (vtk->QueryIntAttribute("enabled", &enabled) == XML_SUCCESS) {
            SetPetscOptionInt("-vtk_output", enabled);
        }
    }

    XMLElement* binary = output->FirstChildElement("binary");
    if (binary) {
        int enabled = 0;
        if (binary->QueryIntAttribute("enabled", &enabled) == XML_SUCCESS) {
            SetPetscOptionInt("-binary", enabled);
        }
    }
}

// Parse grid and boundaries files
static void ParseFiles(XMLElement* root) {
    XMLElement* grid = root->FirstChildElement("grid");
    if (grid) {
        const char* file = grid->Attribute("file");
        if (file) {
            SetPetscOption("-grid", file);
        }
        const char* format = grid->Attribute("format");
        if (format && strcmp(format, "xyz") == 0) {
            SetPetscOptionInt("-xyz", 1);
        }
    }

    XMLElement* boundaries = root->FirstChildElement("boundaries");
    if (boundaries) {
        const char* file = boundaries->Attribute("file");
        if (file) {
            SetPetscOption("-bcs", file);
        }
    }
}

// Parse immersed boundary section
static void ParseImmersedBoundary(XMLElement* root) {
    XMLElement* ibm = root->FirstChildElement("immersed_boundary");
    if (!ibm) return;

    int enabled = 0;
    if (ibm->QueryIntAttribute("enabled", &enabled) == XML_SUCCESS && enabled) {
        SetPetscOptionInt("-imm", 1);
    }

    int bodies = 1;
    if (ibm->QueryIntAttribute("bodies", &bodies) == XML_SUCCESS) {
        SetPetscOptionInt("-body", bodies);
    }
}

// Parse rotor model section
static void ParseRotorModel(XMLElement* root) {
    XMLElement* rotor = root->FirstChildElement("rotor_model");
    if (!rotor) return;

    int enabled = 0;
    if (rotor->QueryIntAttribute("enabled", &enabled) == XML_SUCCESS && enabled) {
        SetPetscOptionInt("-rotor_modeled", enabled);
    }

    int turbines = 1;
    if (rotor->QueryIntAttribute("turbines", &turbines) == XML_SUCCESS) {
        SetPetscOptionInt("-turbine", turbines);
    }

    int blades = 3;
    if (rotor->QueryIntAttribute("blades", &blades) == XML_SUCCESS) {
        SetPetscOptionInt("-num_blade", blades);
    }

    double tip_speed_ratio = 0.0;
    if (rotor->QueryDoubleAttribute("tip_speed_ratio", &tip_speed_ratio) == XML_SUCCESS) {
        SetPetscOptionReal("-tipspeedratio", tip_speed_ratio);
    }
}

// Main parsing function
extern "C" int ParseXMLControlFile(const char* filename) {
    XMLDocument doc;
    XMLError err = doc.LoadFile(filename);

    if (err != XML_SUCCESS) {
        PetscPrintf(PETSC_COMM_WORLD, "Error loading XML file '%s': %s\n",
                    filename, doc.ErrorStr());
        return -1;
    }

    XMLElement* root = doc.FirstChildElement("vfswind");
    if (!root) {
        PetscPrintf(PETSC_COMM_WORLD, "Error: No <vfswind> root element found in '%s'\n", filename);
        return -1;
    }

    // Check version
    const char* version = root->Attribute("version");
    if (version) {
        PetscPrintf(PETSC_COMM_WORLD, "Parsing VFS-Wind XML configuration version %s\n", version);
    }

    // Parse all sections
    ParseSimulation(root);
    ParsePhysics(root);
    ParseTurbulence(root);
    ParseLevelset(root);
    ParseSolvers(root);
    ParseParallel(root);
    ParseOutput(root);
    ParseFiles(root);
    ParseImmersedBoundary(root);
    ParseRotorModel(root);

    PetscPrintf(PETSC_COMM_WORLD, "Successfully parsed XML configuration from '%s'\n", filename);

    return 0;
}

#endif /* ENABLE_XML_INPUT */
