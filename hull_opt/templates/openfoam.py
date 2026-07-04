from pathlib import Path
from jinja2 import Template
import numpy as np
from typing import Optional


TEMPLATES = {}

TEMPLATES["blockMeshDict"] = Template("""/*---------------------------------------------------------------------------*\\
| =========                 |                                                 |
| \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |
|  \\\\    /   O peration     | Version:  v2206                                 |
|   \\\\  /    A nd           | Web:      www.OpenFOAM.com                      |
|    \\\\/     M anipulation  |                                                 |
\\*---------------------------------------------------------------------------*/
FoamFile
{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      blockMeshDict;
}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

convertToMeters 1;

vertices
(
    ({{ -xmax }} {{ -ymax }} {{ -zmin }})
    ({{  xmax }} {{ -ymax }} {{ -zmin }})
    ({{  xmax }} {{  ymax }} {{ -zmin }})
    ({{ -xmax }} {{  ymax }} {{ -zmin }})
    ({{ -xmax }} {{ -ymax }} {{  zmax }})
    ({{  xmax }} {{ -ymax }} {{  zmax }})
    ({{  xmax }} {{  ymax }} {{  zmax }})
    ({{ -xmax }} {{  ymax }} {{  zmax }})
);

blocks
(
    hex (0 1 2 3 4 5 6 7) ({{ nx }} {{ ny }} {{ nz }}) simpleGrading (1 1 1)
);

edges
(
);

boundary
(
    inlet
    {
        type patch;
        faces
        (
            (1 5 4 0)
        );
    }
    outlet
    {
        type patch;
        faces
        (
            (3 7 6 2)
        );
    }
    sides
    {
        type symmetry;
        faces
        (
            (0 4 7 3)
            (2 6 5 1)
        );
    }
    atmosphere
    {
        type patch;
        faces
        (
            (4 5 6 7)
        );
    }
    bottom
    {
        type wall;
        faces
        (
            (0 3 2 1)
        );
    }
);

mergePatchPairs
(
);
""")

TEMPLATES["snappyHexMeshDict"] = Template("""/*---------------------------------------------------------------------------*\\
| =========                 |                                                 |
| \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |
|  \\\\    /   O peration     | Version:  v2206                                 |
|   \\\\  /    A nd           | Web:      www.OpenFOAM.com                      |
|    \\\\/     M anipulation  |                                                 |
\\*---------------------------------------------------------------------------*/
FoamFile
{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      snappyHexMeshDict;
}

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

castellatedMesh true;
snap            true;
addLayers       true;

geometry
{
    hull.stl
    {
        type triSurfaceMesh;
        name hull;
        regions
        {
            patch0
            {
                name hull;
            }
        }
    }
    refinementBox
    {
        type searchableBox;
        min ({{ -LWL * 0.8 }} {{ -B * 0.8 }} {{ -T * 2.0 }});
        max ({{  LWL * 0.8 }} {{  B * 0.8 }} {{  T * 0.5 }});
    }
};

castellatedMeshControls
{
    maxLocalCells {{ max_cells }};
    maxGlobalCells {{ max_cells * 10 }};
    minRefinementCells 10;
    maxLoadUnbalance 0.10;
    nCellsBetweenLevels 3;

    features ();

    refinementSurfaces
    {
        hull
        {
            level ({{ min_surface_level }} {{ max_surface_level }});
            patchInfo
            {
                type wall;
                inGroups (hullGroup);
            }
        }
    }

    resolveFeatureAngle 30;

    refinementRegions
    {
        refinementBox
        {
            mode inside;
            levels (({{ box_level }} {{ box_level }}));
        }
    }

    locationInMesh (0 {{ B * 0.75 }} {{ -T * 0.4 }});
    allowFreeStandingZoneFaces true;
}

snapControls
{
    nSmoothPatch 3;
    tolerance 2.0;
    nSolveIter 30;
    nRelaxIter 5;
    nFeatureSnapIter 10;
    implicitFeatureSnap true;
    explicitFeatureSnap true;
    multiRegionFeatureSnap false;
}

addLayersControls
{
    relativeSizes true;
    layers
    {
        hull
        {
            nSurfaceLayers {{ n_layers }};
        }
    }
    expansionRatio 1.3;
    finalLayerThickness 0.3;
    minThickness 0.001;
    nGrow 0;
    featureAngle 60;
    slipFeatureAngle 30;
    nRelaxIter 5;
    nSmoothSurfaceNormals 1;
    nSmoothNormals 3;
    nSmoothThickness 10;
    maxFaceThicknessRatio 0.5;
    maxThicknessToMedialRatio 0.3;
    minMedialAxisAngle 90;
    nMedialAxisIter 10;
    nBufferCellsNoExtrude 0;
    nLayerIter 50;
}

meshQualityControls
{
    #include "meshQualityDict"
    nSmoothScale 4;
    errorReduction 0.75;
    relaxed
    {
        maxNonOrtho 75;
    }
}

writeFlags
(
    scalarLevels
    layerSets
    layerFields
);

mergeTolerance 1e-6;
""")

TEMPLATES["setFieldsDict"] = Template("""/*---------------------------------------------------------------------------*\\
| =========                 |                                                 |
| \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |
|  \\\\    /   O peration     | Version:  v2206                                 |
|   \\\\  /    A nd           | Web:      www.OpenFOAM.com                      |
|    \\\\/     M anipulation  |                                                 |
\\*---------------------------------------------------------------------------*/
FoamFile
{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      setFieldsDict;
}

defaultFieldValues
(
    volScalarFieldValue alpha.water 0
);

regions
(
    boxToCell
    {
        box ({{ -xmax }} {{ -ymax }} {{ -zmin * 2.0 }}) ({{ xmax }} {{ ymax }} 0);
        fieldValues
        (
            volScalarFieldValue alpha.water 1
        );
    }
);
""")

TEMPLATES["controlDict"] = Template("""/*---------------------------------------------------------------------------*\\
| =========                 |                                                 |
| \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |
|  \\\\    /   O peration     | Version:  v2206                                 |
|   \\\\  /    A nd           | Web:      www.OpenFOAM.com                      |
|    \\\\/     M anipulation  |                                                 |
\\*---------------------------------------------------------------------------*/
FoamFile
{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      controlDict;
}

application {{ solver }};
startFrom       startTime;
startTime       0;
stopAt          endTime;
endTime         {{ end_time }};
deltaT          {{ delta_t }};
writeControl    adjustableRunTime;
writeInterval   {{ write_interval }};
purgeWrite      0;
writeFormat     ascii;
writePrecision  8;
writeCompression off;
timeFormat      general;
timePrecision   8;
runTimeModifiable true;
adjustTimeStep  yes;
maxCo           {{ max_co }};{% if solver == "interFoam" %}
maxAlphaCo      {{ max_alpha_co }};{% endif %}

functions
{
    forces
    {
        type            forces;
        libs            (forces);
        writeControl    timeStep;
        writeInterval   1;
        patches         (hull);
        rho             {% if solver == "simpleFoam" %}rhoInf{% else %}rho{% endif %};
        rhoInf          {{ rho }};
        CofR            (0 0 0);
        log             true;
    }

    {% if six_dof %}
    sixDoF
    {
        type            sixDoFMotion;
        libs            (sixDoFRigidBodyMotion);
        writeControl    timeStep;
        writeInterval   1;
        patches         (hull);
        rho             rho;
        rhoInf          {{ rho }};
        CofR            (0 0 0);
    }
    {% endif %}

    {% if solver == "simpleFoam" %}
    hullPressure
    {
        type            surfaceFieldValue;
        libs            (fieldFunctionObjects);
        operation       max;
        regionType      patch;
        name            hull;
        fields          (p);
        writeControl    writeTime;
        writeInterval   1;
        writeFields     false;
    }
    {% endif %}
};
""")

TEMPLATES["fvSchemes"] = Template("""/*---------------------------------------------------------------------------*\\
| =========                 |                                                 |
| \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |
|  \\\\    /   O peration     | Version:  v2206                                 |
|   \\\\  /    A nd           | Web:      www.OpenFOAM.com                      |
|    \\\\/     M anipulation  |                                                 |
\\*---------------------------------------------------------------------------*/
FoamFile
{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      fvSchemes;
}

ddtSchemes
{
    default         Euler;
}

gradSchemes
{
    default         Gauss linear;
    grad(p)         Gauss linear;
    grad(U)         Gauss linear;
    grad(rho)       Gauss linear;
    grad(alpha)     Gauss linear;
    grad(alpha.water) Gauss linear;
}

divSchemes
{
    {% if solver == "simpleFoam" %}
    div(phi,U)      Gauss linearUpwind limited;
    {% else %}
    div(rhoPhi,U)   Gauss vanLeerV;
    div(phi,alpha)  Gauss vanLeer;
    div(phirb,alpha) Gauss interfaceCompression;
    div(((rho*nuEff)*dev2(T(grad(U))))) Gauss linear;
    {% endif %}
    div(phi,k)      Gauss upwind;
    div(phi,omega)  Gauss upwind;
    div((nuEff*dev2(T(grad(U))))) Gauss linear;
}

laplacianSchemes
{
    default         Gauss linear corrected;
    laplacian(rhoEff,U) Gauss linear corrected;
    laplacian((1|A(U)),p) Gauss linear corrected;
}

interpolationSchemes
{
    default         linear;
}

snGradSchemes
{
    default         corrected;
}

wallDist
{
    method          meshWave;
}
""")

TEMPLATES["fvSolution"] = Template("""/*---------------------------------------------------------------------------*\\
| =========                 |                                                 |
| \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |
|  \\\\    /   O peration     | Version:  v2206                                 |
|   \\\\  /    A nd           | Web:      www.OpenFOAM.com                      |
|    \\\\/     M anipulation  |                                                 |
\\*---------------------------------------------------------------------------*/
FoamFile
{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      fvSolution;
}

solvers
{
    Phi
    {
        solver          PCG;
        preconditioner  DIC;
        tolerance       1e-7;
        relTol          0.01;
    }

    p
    {
        solver          PCG;
        preconditioner  DIC;
        tolerance       1e-7;
        relTol          0.01;
    }

{% if solver == "simpleFoam" %}
    U
    {
        solver          smoothSolver;
        smoother        GaussSeidel;
        tolerance       1e-8;
        relTol          0.1;
        nSweeps         1;
    }

    k
    {
        solver          smoothSolver;
        smoother        GaussSeidel;
        tolerance       1e-8;
        relTol          0.1;
        nSweeps         1;
    }

    omega
    {
        solver          smoothSolver;
        smoother        GaussSeidel;
        tolerance       1e-8;
        relTol          0.1;
        nSweeps         1;
    }
{% else %}
    p_rgh
    {
        $p;
        tolerance       1e-8;
        relTol          0.001;
    }

    p_rghFinal
    {
        $p_rgh;
        relTol          0;
    }

    pcorr
    {
        $p;
        relTol          0;
    }

    pcorrFinal
    {
        $p;
        relTol          0;
    }

    "(U|k|omega)"
    {
        solver          smoothSolver;
        smoother        GaussSeidel;
        tolerance       1e-8;
        relTol          0.1;
        nSweeps         1;
    }

    "(U|k|omega)Final"
    {
        $U;
        relTol          0;
    }

    "alpha.*"
    {
        solver          smoothSolver;
        smoother        GaussSeidel;
        tolerance       1e-8;
        relTol          0;
        nSweeps         1;
        nAlphaCorr      2;
        nAlphaSubCycles 2;
        cAlpha          1;
    }
{% endif %}
}

potentialFlow
{
    PhiRefCell   0;
    PhiRefValue  0;
}

pRefCell      0;
pRefValue     0;

{% if solver == "simpleFoam" %}
SIMPLE
{
    nNonOrthogonalCorrectors 0;
    consistent yes;
    pRefCell      0;
    pRefValue     0;
}

relaxationFactors
{
    fields
    {
        p               0.3;
    }
    equations
    {
        ".*"            0.7;
    }
}
{% else %}
PIMPLE
{
    nCorrectors      3;
    nNonOrthogonalCorrectors 1;
    nOuterCorrectors 3;
    momentumPredictor yes;
    transonic       no;
    maxIter         100;
    pRefCell        0;
    pRefValue       0;
}

relaxationFactors
{
    equations
    {
        ".*"           1;
    }
}
{% endif %}
""")

TEMPLATES["transportProperties_simpleFoam"] = Template("""/*---------------------------------------------------------------------------*\\
| =========                 |                                                 |
| \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |
|  \\\\    /   O peration     | Version:  v2206                                 |
|   \\\\  /    A nd           | Web:      www.OpenFOAM.com                      |
|    \\\\/     M anipulation  |                                                 |
\\*---------------------------------------------------------------------------*/
FoamFile
{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      transportProperties;
}

transportModel  Newtonian;
nu              {{ nu_water }};
rho             {{ rho_water }};
""")

TEMPLATES["transportProperties"] = Template("""/*---------------------------------------------------------------------------*\\
| =========                 |                                                 |
| \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |
|  \\\\    /   O peration     | Version:  v2206                                 |
|   \\\\  /    A nd           | Web:      www.OpenFOAM.com                      |
|    \\\\/     M anipulation  |                                                 |
\\*---------------------------------------------------------------------------*/
FoamFile
{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      transportProperties;
}

phases (water air);

water
{
    transportModel  Newtonian;
    nu              {{ nu_water }};
    rho             {{ rho_water }};
}

air
{
    transportModel  Newtonian;
    nu              1.48e-5;
    rho             1.0;
}

sigma           0.07;
""")

TEMPLATES["turbulenceProperties"] = Template("""/*---------------------------------------------------------------------------*\\
| =========                 |                                                 |
| \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |
|  \\\\    /   O peration     | Version:  v2206                                 |
|   \\\\  /    A nd           | Web:      www.OpenFOAM.com                      |
|    \\\\/     M anipulation  |                                                 |
\\*---------------------------------------------------------------------------*/
FoamFile
{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      turbulenceProperties;
}

simulationType  RAS;

RAS
{
    RASModel        kOmegaSST;
    turbulence      on;
    printCoeffs     on;
}
""")

TEMPLATES["g"] = Template("""/*---------------------------------------------------------------------------*\\
| =========                 |                                                 |
| \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |
|  \\\\    /   O peration     | Version:  v2206                                 |
|   \\\\  /    A nd           | Web:      www.OpenFOAM.com                      |
|    \\\\/     M anipulation  |                                                 |
\\*---------------------------------------------------------------------------*/
FoamFile
{
    version     2.0;
    format      ascii;
    class       uniformDimensionedVectorField;
    object      g;
}

dimensions      [0 1 -2 0 0 0 0];
value           (0 0 {{ -gravity }});
""")

TEMPLATES["U"] = Template("""/*---------------------------------------------------------------------------*\\
| =========                 |                                                 |
| \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |
|  \\\\    /   O peration     | Version:  v2206                                 |
|   \\\\  /    A nd           | Web:      www.OpenFOAM.com                      |
|    \\\\/     M anipulation  |                                                 |
\\*---------------------------------------------------------------------------*/
FoamFile
{
    version     2.0;
    format      ascii;
    class       volVectorField;
    object      U;
}

dimensions      [0 1 -1 0 0 0 0];

internalField   uniform (0 0 0);

boundaryField
{
    inlet
    {
        type            fixedValue;
        value           uniform ({{ speed_ms }} 0 0);
    }
    outlet
    {
        type            zeroGradient;
    }
    sides
    {
        type            symmetry;
    }
    atmosphere
    {
        type            pressureInletOutletVelocity;
        value           uniform (0 0 0);
    }
    bottom
    {
        type            noSlip;
    }
    hull
    {
        type            noSlip;
    }
}
""")

TEMPLATES["p_rgh"] = Template("""/*---------------------------------------------------------------------------*\\
| =========                 |                                                 |
| \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |
|  \\\\    /   O peration     | Version:  v2206                                 |
|   \\\\  /    A nd           | Web:      www.OpenFOAM.com                      |
|    \\\\/     M anipulation  |                                                 |
\\*---------------------------------------------------------------------------*/
FoamFile
{
    version     2.0;
    format      ascii;
    class       volScalarField;
    object      p_rgh;
}

dimensions      [1 -1 -2 0 0 0 0];

internalField   uniform 0;

boundaryField
{
    inlet
    {
        type            fixedFluxPressure;
    }
    outlet
    {
        type            fixedFluxPressure;
    }
    sides
    {
        type            symmetry;
    }
    atmosphere
    {
        type            totalPressure;
        p0              uniform 0;
    }
    bottom
    {
        type            fixedFluxPressure;
    }
    hull
    {
        type            fixedFluxPressure;
    }
}
""")

TEMPLATES["alpha_water"] = Template("""/*---------------------------------------------------------------------------*\\
| =========                 |                                                 |
| \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |
|  \\\\    /   O peration     | Version:  v2206                                 |
|   \\\\  /    A nd           | Web:      www.OpenFOAM.com                      |
|    \\\\/     M anipulation  |                                                 |
\\*---------------------------------------------------------------------------*/
FoamFile
{
    version     2.0;
    format      ascii;
    class       volScalarField;
    object      alpha.water;
}

dimensions      [0 0 0 0 0 0 0];

internalField   uniform 0;

boundaryField
{
    inlet
    {
        type            fixedValue;
        value           uniform 0;
    }
    outlet
    {
        type            zeroGradient;
    }
    sides
    {
        type            symmetry;
    }
    atmosphere
    {
        type            inletOutlet;
        inletValue      uniform 0;
        value           uniform 0;
    }
    bottom
    {
        type            zeroGradient;
    }
    hull
    {
        type            zeroGradient;
    }
}
""")

TEMPLATES["k"] = Template("""/*---------------------------------------------------------------------------*\\
| =========                 |                                                 |
| \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |
|  \\\\    /   O peration     | Version:  v2206                                 |
|   \\\\  /    A nd           | Web:      www.OpenFOAM.com                      |
|    \\\\/     M anipulation  |                                                 |
\\*---------------------------------------------------------------------------*/
FoamFile
{
    version     2.0;
    format      ascii;
    class       volScalarField;
    object      k;
}

dimensions      [0 2 -2 0 0 0 0];

internalField   uniform {{ k_value }};

boundaryField
{
    inlet
    {
        type            fixedValue;
        value           uniform {{ k_value }};
    }
    outlet
    {
        type            zeroGradient;
    }
    sides
    {
        type            symmetry;
    }
    atmosphere
    {
        type            inletOutlet;
        inletValue      uniform {{ k_value }};
        value           uniform {{ k_value }};
    }
    bottom
    {
        type            kqRWallFunction;
        value           uniform {{ k_value }};
    }
    hull
    {
        type            kqRWallFunction;
        value           uniform {{ k_value }};
    }
}
""")

TEMPLATES["nut"] = Template("""/*---------------------------------------------------------------------------*\\
| =========                 |                                                 |
| \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |
|  \\\\    /   O peration     | Version:  v2206                                 |
|   \\\\  /    A nd           | Web:      www.OpenFOAM.com                      |
|    \\\\/     M anipulation  |                                                 |
\\*---------------------------------------------------------------------------*/
FoamFile
{
    version     2.0;
    format      ascii;
    class       volScalarField;
    object      nut;
}

dimensions      [0 2 -1 0 0 0 0];

internalField   uniform {{ k_value / omega_value }};

boundaryField
{
    inlet
    {
        type            fixedValue;
        value           uniform {{ k_value / omega_value }};
    }
    outlet
    {
        type            zeroGradient;
    }
    sides
    {
        type            symmetry;
    }
    atmosphere
    {
        type            inletOutlet;
        inletValue      uniform {{ k_value / omega_value }};
        value           uniform {{ k_value / omega_value }};
    }
    bottom
    {
        type            nutkWallFunction;
        value           uniform {{ k_value / omega_value }};
    }
    hull
    {
        type            nutkWallFunction;
        value           uniform {{ k_value / omega_value }};
    }
}
""")

TEMPLATES["pointDisplacement"] = Template("""/*---------------------------------------------------------------------------*\\
| =========                 |                                                 |
| \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |
|  \\\\    /   O peration     | Version:  v2206                                 |
|   \\\\  /    A nd           | Web:      www.OpenFOAM.com                      |
|    \\\\/     M anipulation  |                                                 |
\\*---------------------------------------------------------------------------*/
FoamFile
{
    version     2.0;
    format      ascii;
    class       pointVectorField;
    object      pointDisplacement;
}

dimensions      [0 1 0 0 0 0 0];

internalField   uniform (0 0 0);

boundaryField
{
    inlet
    {
        type            fixedValue;
        value           uniform (0 0 0);
    }
    outlet
    {
        type            fixedValue;
        value           uniform (0 0 0);
    }
    sides
    {
        type            symmetry;
    }
    atmosphere
    {
        type            fixedValue;
        value           uniform (0 0 0);
    }
    bottom
    {
        type            fixedValue;
        value           uniform (0 0 0);
    }
    hull
    {
        type            calculated;
        value           uniform (0 0 0);
    }
}
""")

TEMPLATES["p"] = Template("""/*---------------------------------------------------------------------------*\\
| =========                 |                                                 |
| \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |
|  \\\\    /   O peration     | Version:  v2206                                 |
|   \\\\  /    A nd           | Web:      www.OpenFOAM.com                      |
|    \\\\/     M anipulation  |                                                 |
\\*---------------------------------------------------------------------------*/
FoamFile
{
    version     2.0;
    format      ascii;
    class       volScalarField;
    object      p;
}

dimensions      [0 2 -2 0 0 0 0];

internalField   uniform 0;

boundaryField
{
    inlet
    {
        type            zeroGradient;
    }
    outlet
    {
        type            zeroGradient;
    }
    sides
    {
        type            symmetry;
    }
    atmosphere
    {
        type            totalPressure;
        p0              uniform 0;
    }
    bottom
    {
        type            zeroGradient;
    }
    hull
    {
        type            zeroGradient;
    }
}
""")

TEMPLATES["dynamicMeshDict"] = Template("""/*---------------------------------------------------------------------------*\\
| =========                 |                                                 |
| \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |
|  \\\\    /   O peration     | Version:  v2206                                 |
|   \\\\  /    A nd           | Web:      www.OpenFOAM.com                      |
|    \\\\/     M anipulation  |                                                 |
\\*---------------------------------------------------------------------------*/
FoamFile
{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      dynamicMeshDict;
}

dynamicFvMesh   dynamicMotionSolverFvMesh;

motionSolver   sixDoFRigidBodyMotion;

sixDoFRigidBodyMotionCoeffs
{
    patches         (hull);
    rho             rho;
    rhoInf          {{ rho }};
    g               (0 0 -{{ gravity }});
    centreOfMass    (0 0 {{ cg_z }});
    mass            {{ mass }};
    momentOfInertia ({{ Ixx }} {{ Iyy }} {{ Izz }});
    solver
    {
        type    Newmark;
    }
    innerDistance   {{ T * 0.5 }};
    outerDistance   {{ T * 2.0 }};
    report          on;
    accelerationRelaxation 0.5;
    nPredictor 2;
    rampDuration   0.01;
{% if initial_state %}
{% if 'velocity' in initial_state %}
    velocity        ({{ initial_state.velocity[0] }} {{ initial_state.velocity[1] }} {{ initial_state.velocity[2] }});
{% endif %}
{% if 'angularMomentum' in initial_state %}
    angularMomentum ({{ initial_state.angularMomentum[0] }} {{ initial_state.angularMomentum[1] }} {{ initial_state.angularMomentum[2] }});
{% endif %}
{% endif %}
}

motionSolverLibs (sixDoFRigidBodyMotion);
""")

TEMPLATES["omega"] = Template("""/*---------------------------------------------------------------------------*\\
| =========                 |                                                 |
| \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |
|  \\\\    /   O peration     | Version:  v2206                                 |
|   \\\\  /    A nd           | Web:      www.OpenFOAM.com                      |
|    \\\\/     M anipulation  |                                                 |
\\*---------------------------------------------------------------------------*/
FoamFile
{
    version     2.0;
    format      ascii;
    class       volScalarField;
    object      omega;
}

dimensions      [0 0 -1 0 0 0 0];

internalField   uniform {{ omega_value }};

boundaryField
{
    inlet
    {
        type            fixedValue;
        value           uniform {{ omega_value }};
    }
    outlet
    {
        type            zeroGradient;
    }
    sides
    {
        type            symmetry;
    }
    atmosphere
    {
        type            inletOutlet;
        inletValue      uniform {{ omega_value }};
        value           uniform {{ omega_value }};
    }
    bottom
    {
        type            omegaWallFunction;
        value           uniform {{ omega_value }};
    }
    hull
    {
        type            omegaWallFunction;
        value           uniform {{ omega_value }};
    }
}
""")

TEMPLATES["decomposeParDict"] = Template("""/*---------------------------------------------------------------------------*\\
| =========                 |                                                 |
| \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |
|  \\\\    /   O peration     | Version:  v2206                                 |
|   \\\\  /    A nd           | Web:      www.OpenFOAM.com                      |
|    \\\\/     M anipulation  |                                                 |
\\*---------------------------------------------------------------------------*/
FoamFile
{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      decomposeParDict;
}

numberOfSubdomains {{ n_procs }};

method          scotch;
""")


def write_openfoam_case(case_dir: Path, stl_path: str, speed_ms: float,
                        LWL: float, B: float, T: float,
                        rho: float = 1025.0, nu: float = 1.19e-6,
                        gravity: float = 9.81,
                        mesh_levels: tuple = (2, 3), n_layers: int = 3,
                        solver: str = "interFoam",
                        six_dof: bool = False,
                        end_time: float = 10.0, delta_t: float = 0.001,
                        write_interval: float = 0.1,
                        max_cells: int = 200000,
                        mass: float = 0.0, cg_z: float = -0.1,
                        Ixx: float = 0.0, Iyy: float = 0.0, Izz: float = 0.0,
                        initial_state: Optional[dict] = None,
                        max_co: float = 0.5, max_alpha_co: float = 0.5):
    case_dir = Path(case_dir)
    (case_dir / "constant").mkdir(parents=True, exist_ok=True)
    (case_dir / "system").mkdir(parents=True, exist_ok=True)
    (case_dir / "0").mkdir(parents=True, exist_ok=True)

    # copy STL
    import shutil
    stl_dest = case_dir / "constant" / "triSurface" / "hull.stl"
    stl_dest.parent.mkdir(exist_ok=True)
    shutil.copy2(stl_path, stl_dest)

    if six_dof and mass == 0.0:
        mass = rho * B * LWL * T * 0.5
    if six_dof and Ixx == 0.0:
        Ixx = mass * (B ** 2 + T ** 2) / 12.0
        Iyy = mass * (LWL ** 2 + T ** 2) / 12.0
        Izz = mass * (LWL ** 2 + B ** 2) / 12.0

    xmax = 3.0 * LWL
    ymax = 2.0 * B
    zmin = 2.0 * T
    zmax = 2.0 * T
    nx = int(np.ceil(max_cells ** (1/3) * xmax / (xmax + ymax + zmax + zmin)))
    ny = int(np.ceil(max_cells ** (1/3) * ymax / (xmax + ymax + zmax + zmin)))
    nz = int(np.ceil(max_cells ** (1/3) * (zmax + zmin) / (xmax + ymax + zmax + zmin)))
    nx = max(10, nx)
    ny = max(5, ny)
    nz = max(5, nz)

    if speed_ms > 0:
        turbulence_intensity = 0.05
        k_val = 1.5 * (speed_ms * turbulence_intensity) ** 2
        omega_val = np.sqrt(k_val) / (0.07 * LWL)
    else:
        # Minimum turbulent values for zero-speed cases (drop impact, etc.)
        k_val = 1e-8
        omega_val = 1.0

    replacements = {
        "xmax": xmax, "ymax": ymax, "zmin": zmin, "zmax": zmax,
        "nx": nx, "ny": ny, "nz": nz,
        "LWL": LWL, "B": B, "T": T,
        "max_cells": max_cells,
        "min_surface_level": mesh_levels[0],
        "max_surface_level": mesh_levels[1],
        "feature_level": mesh_levels[0],
        "box_level": mesh_levels[0] + 1,
        "n_layers": n_layers,
        "solver": solver,
        "end_time": end_time, "delta_t": delta_t, "write_interval": write_interval,
        "rho": rho, "rho_water": rho, "nu_water": nu,
        "gravity": gravity,
        "speed_ms": speed_ms,
        "k_value": k_val, "omega_value": omega_val,
        "six_dof": six_dof,
        "mass": mass, "cg_z": cg_z, "Ixx": Ixx, "Iyy": Iyy, "Izz": Izz,
        "initial_state": initial_state if initial_state else None,
        "max_co": max_co, "max_alpha_co": max_alpha_co,
    }

    files = {
        "system/blockMeshDict": "blockMeshDict",
        "system/snappyHexMeshDict": "snappyHexMeshDict",
        "system/controlDict": "controlDict",
        "system/fvSchemes": "fvSchemes",
        "system/fvSolution": "fvSolution",
        "constant/transportProperties": "transportProperties",
        "constant/turbulenceProperties": "turbulenceProperties",
        "constant/g": "g",
        "0/U": "U",
        "0/k": "k",
        "0/omega": "omega",
        "0/nut": "nut",
    }

    if solver == "simpleFoam":
        files["0/p"] = "p"
        files["constant/transportProperties"] = "transportProperties_simpleFoam"
    else:
        files["0/p_rgh"] = "p_rgh"
        files["0/alpha.water"] = "alpha_water"
        files["system/setFieldsDict"] = "setFieldsDict"

    if six_dof:
        files["constant/dynamicMeshDict"] = "dynamicMeshDict"
        files["0/pointDisplacement"] = "pointDisplacement"

    for rel_path, template_key in files.items():
        output_path = case_dir / rel_path
        content = TEMPLATES[template_key].render(**replacements)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            f.write(content)

    import os as _os
    _wm_dir = _os.environ.get("WM_PROJECT_DIR", "")
    _mqd_candidates = [
        Path(_wm_dir) / "etc" / "caseDicts" / "meshQualityDict",
        Path("/usr/lib/openfoam/openfoam2512/etc/caseDicts/meshQualityDict"),
        Path("/opt/openfoam2512/etc/caseDicts/meshQualityDict"),
    ]
    mesh_quality_dict = next((p for p in _mqd_candidates if p.exists()), None)
    if mesh_quality_dict is not None:
        shutil.copy2(mesh_quality_dict, case_dir / "system" / "meshQualityDict")
