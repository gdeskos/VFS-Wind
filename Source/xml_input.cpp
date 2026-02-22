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
            SetPetscOptionInt("-totalsteps", totalsteps);
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

    XMLElement* options = sim->FirstChildElement("options");
    if (options) {
        int delete_output = 0;
        int second_order = 0;
        int laplacian = 0;
        int averaging = 0;

        if (options->QueryIntAttribute("delete_previous", &delete_output) == XML_SUCCESS) {
            SetPetscOptionInt("-delete", delete_output);
        }
        if (options->QueryIntAttribute("second_order", &second_order) == XML_SUCCESS) {
            SetPetscOptionInt("-second_order", second_order);
        }
        if (options->QueryIntAttribute("laplacian", &laplacian) == XML_SUCCESS) {
            SetPetscOptionInt("-laplacian", laplacian);
        }
        if (options->QueryIntAttribute("averaging", &averaging) == XML_SUCCESS) {
            SetPetscOptionInt("-averaging", averaging);
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
        SetPetscOptionReal("-gx", gx);
        SetPetscOptionReal("-gy", gy);
        SetPetscOptionReal("-gz", gz);
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
            // Note: "cs" attribute is parsed but not set as PETSc option because
            // -cs is used by post-processors (data.c) as an integer flag.
            // The Smagorinsky constant is computed dynamically in les.c.
            double max_cs = 0.2;
            if (les->QueryDoubleAttribute("max_cs", &max_cs) == XML_SUCCESS) {
                SetPetscOptionReal("-max_cs", max_cs);
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
            SetPetscOptionInt("-viscosity_wallmodel", 1);
        }
        double channel_height = 0.0;
        if (wf->QueryDoubleAttribute("channel_height", &channel_height) == XML_SUCCESS) {
            SetPetscOptionReal("-channel_height", channel_height);
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

    // Fluid properties (fluid0 = water/heavy, fluid1 = air/light)
    XMLElement* fluid0 = ls->FirstChildElement("fluid0");
    if (fluid0) {
        double density = 0.0, viscosity = 0.0;
        if (fluid0->QueryDoubleAttribute("density", &density) == XML_SUCCESS) {
            SetPetscOptionReal("-rho0", density);
        }
        if (fluid0->QueryDoubleAttribute("viscosity", &viscosity) == XML_SUCCESS) {
            SetPetscOptionReal("-mu0", viscosity);
        }
    }

    XMLElement* fluid1 = ls->FirstChildElement("fluid1");
    if (fluid1) {
        double density = 0.0, viscosity = 0.0;
        if (fluid1->QueryDoubleAttribute("density", &density) == XML_SUCCESS) {
            SetPetscOptionReal("-rho1", density);
        }
        if (fluid1->QueryDoubleAttribute("viscosity", &viscosity) == XML_SUCCESS) {
            SetPetscOptionReal("-mu1", viscosity);
        }
    }

    // Levelset iterations
    XMLElement* iterations = ls->FirstChildElement("iterations");
    if (iterations) {
        int it = 10;
        if (iterations->QueryIntText(&it) == XML_SUCCESS) {
            SetPetscOptionInt("-levelset_it", it);
        }
    }

    // Sloshing mode
    int sloshing = 0;
    if (ls->QueryIntAttribute("sloshing", &sloshing) == XML_SUCCESS && sloshing > 0) {
        SetPetscOptionInt("-sloshing", sloshing);
    }

    // Surface tension
    int stension = 0;
    if (ls->QueryIntAttribute("surface_tension", &stension) == XML_SUCCESS) {
        SetPetscOptionInt("-stension", stension);
    }

    // Level initialization
    int level_in = 0;
    if (ls->QueryIntAttribute("level_in", &level_in) == XML_SUCCESS) {
        SetPetscOptionInt("-level_in", level_in);
    }

    double level_in_height = 0.0;
    if (ls->QueryDoubleAttribute("level_in_height", &level_in_height) == XML_SUCCESS) {
        SetPetscOptionReal("-level_in_height", level_in_height);
    }

    // Interface thickness
    double dthick = 0.0;
    if (ls->QueryDoubleAttribute("dthick", &dthick) == XML_SUCCESS) {
        SetPetscOptionReal("-dthick", dthick);
    }

    // Levelset tau (reinitialization parameter)
    double levelset_tau = 0.0;
    if (ls->QueryDoubleAttribute("tau", &levelset_tau) == XML_SUCCESS) {
        SetPetscOptionReal("-levelset_tau", levelset_tau);
    }

    // Subdt for levelset
    int subdt = 0;
    if (ls->QueryIntAttribute("subdt", &subdt) == XML_SUCCESS) {
        SetPetscOptionInt("-subdt_levelset", subdt);
    }

    // Air flow levelset
    int air_flow = 0;
    if (ls->QueryIntAttribute("air_flow", &air_flow) == XML_SUCCESS) {
        SetPetscOptionInt("-air_flow_levelset", air_flow);
    }

    // Fix level
    int fix_level = 0;
    if (ls->QueryIntAttribute("fix_level", &fix_level) == XML_SUCCESS) {
        SetPetscOptionInt("-fix_level", fix_level);
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
        SetPetscOptionReal("-poisson_tol", tolerance);
    }

    XMLElement* momentum = solvers->FirstChildElement("momentum");
    if (momentum) {
        int implicit = 0;
        int max_iterations = 50;
        double tolerance = 1.e-5;

        momentum->QueryIntAttribute("implicit", &implicit);
        momentum->QueryIntAttribute("max_iterations", &max_iterations);
        momentum->QueryDoubleAttribute("tolerance", &tolerance);

        if (implicit > 0) {
            SetPetscOptionInt("-imp", implicit);
            SetPetscOptionInt("-imp_MAX_IT", max_iterations);
            SetPetscOptionReal("-imp_tol", tolerance);
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
        int ii = 0, jj = 0, kk = 0;
        periodic->QueryIntAttribute("i", &i);
        periodic->QueryIntAttribute("j", &j);
        periodic->QueryIntAttribute("k", &k);
        // Also support ii, jj, kk variants (used in channel flow)
        periodic->QueryIntAttribute("ii", &ii);
        periodic->QueryIntAttribute("jj", &jj);
        periodic->QueryIntAttribute("kk", &kk);

        if (i) SetPetscOptionInt("-i_periodic", i);
        if (j) SetPetscOptionInt("-j_periodic", j);
        if (k) SetPetscOptionInt("-k_periodic", k);
        if (ii) SetPetscOptionInt("-ii_periodic", ii);
        if (jj) SetPetscOptionInt("-jj_periodic", jj);
        if (kk) SetPetscOptionInt("-kk_periodic", kk);
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

// Parse FSI (Fluid-Structure Interaction) section
static void ParseFSI(XMLElement* root) {
    XMLElement* fsi = root->FirstChildElement("fsi");
    if (!fsi) return;

    int enabled = 0;
    if (fsi->QueryIntAttribute("enabled", &enabled) == XML_SUCCESS && enabled) {
        SetPetscOptionInt("-fsi", 1);
    }

    // Forced motion mode
    int forced_motion = 0;
    if (fsi->QueryIntAttribute("forced_motion", &forced_motion) == XML_SUCCESS) {
        SetPetscOptionInt("-forced_motion", forced_motion);
    }

    // Fall cylinder case
    int fall_cylinder = 0;
    if (fsi->QueryIntAttribute("fall_cylinder", &fall_cylinder) == XML_SUCCESS) {
        SetPetscOptionInt("-fall_cyll_case", fall_cylinder);
    }

    // Center position
    XMLElement* center = fsi->FirstChildElement("center");
    if (center) {
        double x = 0.0, y = 0.0, z = 0.0;
        center->QueryDoubleAttribute("x", &x);
        center->QueryDoubleAttribute("y", &y);
        center->QueryDoubleAttribute("z", &z);
        SetPetscOptionReal("-x_c", x);
        SetPetscOptionReal("-y_c", y);
        SetPetscOptionReal("-z_c", z);
    }

    // Degrees of freedom
    XMLElement* dof = fsi->FirstChildElement("dof");
    if (dof) {
        int x = 0, y = 0, z = 0;
        int ax = 0, ay = 0, az = 0;
        dof->QueryIntAttribute("x", &x);
        dof->QueryIntAttribute("y", &y);
        dof->QueryIntAttribute("z", &z);
        // Also support acceleration DOF (ax, ay, az) for falling cylinder
        dof->QueryIntAttribute("ax", &ax);
        dof->QueryIntAttribute("ay", &ay);
        dof->QueryIntAttribute("az", &az);
        SetPetscOptionInt("-dgf_x", x);
        SetPetscOptionInt("-dgf_y", y);
        SetPetscOptionInt("-dgf_z", z);
    }

    // Structural parameters
    XMLElement* params = fsi->FirstChildElement("parameters");
    if (params) {
        double red_vel = 0.0;
        if (params->QueryDoubleAttribute("reduced_velocity", &red_vel) == XML_SUCCESS) {
            SetPetscOptionReal("-red_vel", red_vel);
        }

        double damping = 0.0;
        if (params->QueryDoubleAttribute("damping", &damping) == XML_SUCCESS) {
            SetPetscOptionReal("-damp", damping);
        }

        double mass_ratio = 0.0;
        if (params->QueryDoubleAttribute("mass_ratio", &mass_ratio) == XML_SUCCESS) {
            SetPetscOptionReal("-mu_s", mass_ratio);
        }
    }
}

// Parse channel flow section
static void ParseChannelFlow(XMLElement* root) {
    XMLElement* channel = root->FirstChildElement("channel_flow");
    if (!channel) return;

    int enabled = 0;
    if (channel->QueryIntAttribute("enabled", &enabled) == XML_SUCCESS && enabled) {
        double flux = 0.0;
        if (channel->QueryDoubleAttribute("flux", &flux) == XML_SUCCESS) {
            SetPetscOptionReal("-flux", flux);
        }

        int inlet = 1;
        if (channel->QueryIntAttribute("inlet", &inlet) == XML_SUCCESS) {
            SetPetscOptionInt("-inlet", inlet);
        }

        int perturb = 0;
        if (channel->QueryIntAttribute("perturb", &perturb) == XML_SUCCESS) {
            SetPetscOptionInt("-perturb", perturb);
        }
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
    ParseFSI(root);
    ParseRotorModel(root);
    ParseChannelFlow(root);

    PetscPrintf(PETSC_COMM_WORLD, "Successfully parsed XML configuration from '%s'\n", filename);

    return 0;
}

#endif /* ENABLE_XML_INPUT */
