import logging
from pathlib import Path
from typing import Optional
from jinja2 import Template


logger = logging.getLogger(__name__)

TEMPLATES = {}

# GenCase boxfill only generates the drawbox faces that sit well inside the
# definition box: any face flush with (or within ~0.12 m of) the definition
# limits is dropped, silently leaving the tank open (verified empirically:
# bottom|front|back|right produced only 3 walls; +y/+x/+z walls appeared once
# the definition box was enlarged by 0.35 m on every side).  All tank
# drawboxes must therefore keep at least this margin to the definition box.
DEF_MARGIN = 0.35


# ── Focused wave group case ──────────────────────────────────────────────
TEMPLATES["FOCUSED_WAVE_XML"] = FOCUSED_WAVE_XML = Template("""<?xml version="1.0" encoding="UTF-8"?>
<case>
    <casedef>
        <constantsdef>
            <lattice bound="2" fluid="1" comment="Type of lattice to create the initial particles (default=1)" />
            <gravity x="0" y="0" z="-{{ gravity }}" comment="Gravitational acceleration" units_comment="m/s^2" />
            <rhop0 value="{{ rho }}" comment="Reference density of the fluid" units_comment="kg/m^3" />
            <rhopgradient value="2" comment="Initial density gradient 1:Rhop0, 2:Water column, 3:Max. water height (default=2)" />
            <hswl value="0" auto="true" comment="Maximum still water level to calculate speedofsound using coefsound" units_comment="metres (m)" />
            <gamma value="7.0" comment="Polytropic constant for water used in the state equation" />
            <speedsystem value="0" auto="true" comment="Maximum system speed (by default the dam-break propagation is used)" />
            <coefsound value="20" comment="Coefficient to multiply speedsystem" />
            <speedsound value="0" auto="true" comment="Speed of sound to use in the simulation (by default speedofsound=coefsound*speedsystem)" />
            <hdp value="2" comment="Alternative option to calculate the smoothing length (h=hdp*dp)" />
            <cflnumber value="0.2" comment="Coefficient to multiply dt" />
        </constantsdef>
        <mkconfig boundcount="100" fluidcount="9" />
        <geometry>
            <definition dp="{{ dp }}" comment="Initial inter-particle distance" units_comment="metres (m)">
                <pointmin x="-{{ xmax + def_margin }}" y="-{{ ymax + def_margin }}" z="-{{ zmin + def_margin }}" />
                <pointmax x="{{ xmax + def_margin }}" y="{{ ymax + def_margin }}" z="{{ zmax + def_margin }}" />
            </definition>
            <commands>
                <mainlist>
                    <setshapemode>actual | bound</setshapemode>
                    <setdrawmode mode="full" />
                    <setmkbound mk="10" />
                    <drawbox cmt="Wave paddle (mk=10, driven by wavepaddles)">
                        <boxfill>solid</boxfill>
                        <point x="{{ x_paddle_lo }}" y="-{{ ymax }}" z="-{{ zmin }}" />
                        <size x="{{ x_paddle_th }}" y="{{ 2*ymax }}" z="{{ zmax + zmin }}" />
                    </drawbox>
                    <setmkbound mk="0" />
                    <drawbox cmt="Tank: fully closed box (all 6 walls)">
                        <boxfill>all</boxfill>
                        <point x="-{{ xmax }}" y="-{{ ymax }}" z="-{{ zmin }}" />
                        <size x="{{ 2*xmax }}" y="{{ 2*ymax }}" z="{{ zmax + zmin }}" />
                    </drawbox>
                    <setmkbound mk="1" />
                    <drawfilestl file="{{ hull_stl_path }}">
                        <drawmove x="{{ hull_x }}" y="0" z="{{ hull_z }}" />
                    </drawfilestl>
                    <setmkfluid mk="0" />
                    <setboxlimitmode mode="full" />
                    <fillbox x="{{ seed_x }}" y="{{ seed_y }}" z="{{ seed_z }}">
                        <modefill>void</modefill>
                        <point x="-{{ xdomain }}" y="-{{ ydomain }}" z="-{{ zmin }}" />
                        <size x="{{ 2*xdomain }}" y="{{ 2*ydomain }}" z="{{ still_water_level + zmin }}" />
                    </fillbox>
                    <shapeout file="" />
                </mainlist>
            </commands>
        </geometry>
        <motion>
            <objreal ref="10">
                <begin mov="1" start="0" />
                <mvnull id="1" />
            </objreal>
        </motion>
        <floatings>
            <floating mkbound="1">
                <massbody value="{{ mass }}" />
                <inertia x="{{ Ixx }}" y="{{ Iyy }}" z="{{ Izz }}" />
                {{ cg_str }}
            </floating>
        </floatings>
        <initials>
            <densitydepth mkfluid="0" />
        </initials>
    </casedef>
    <execution>
        <special>
            <wavepaddles>
                <piston_focused>
                    <mkbound value="10" />
                    <depth value="{{ water_depth }}" />
                    <start value="0" />
                    <duration value="0" />
                    <pistondir x="1" y="0" z="0" />
                    <spectrum value="jonswap" />
                    <discretization value="cosstretched" />
                    <randomseed value="2" />
                    <waveorder value="2" />
                    <waveheight value="{{ wave_height }}" />
                    <waveperiod value="{{ wave_period }}" />
                    <gainstroke value="1" />
                    <peakcoef value="3.3" />
                    <waves value="1024" />
                    <xf value="{{ xf }}" />
                    <fphase value="0" />
                    <maxwaveh nwaves="1000" />
                    <ramptime value="{{ ramp_time }}" />
                    <fpretime value="0" />
                    <fmovtime value="{{ sim_time }}" />
                    <fmovramp value="{{ ramp_time }}" />
                </piston_focused>
            </wavepaddles>
            <damping>
                <dampingzone>
                    <limitmin x="{{ x_damp }}" y="0" z="0" />
                    <limitmax x="{{ xmax }}" y="0" z="0" />
                    <overlimit value="1" />
                    <redumax value="10" />
                </dampingzone>
            </damping>
        </special>
        <constants>
            <data2d value="false"/>
            <gravity x="0" y="0" z="-{{ gravity }}"/>
            <cflnumber value="0.2"/>
            <gamma value="7.0"/>
            <rhop0 value="{{ rho }}"/>
            <eps value="0"/>
            <dp value="{{ dp }}"/>
            <h value="0"/>
            <b value="0"/>
            <massbound value="0"/>
            <massfluid value="0"/>
        </constants>
        <parameters>
            <parameter key="StepAlgorithm" value="2" comment="Step Algorithm 1:Verlet, 2:Symplectic (default=1)" />
            <parameter key="VerletSteps" value="50" comment="Verlet only: Number of steps to apply Euler timestepping (default=50)" />
            <parameter key="Kernel" value="2" comment="Interaction Kernel 1:Cubic Spline, 2:Wendland (default=2)" />
            <parameter key="ViscoTreatment" value="3" comment="Viscosity formulation 1:Artificial, 2:Laminar+SPS, 3:Laminar (default=1)" />
            <parameter key="Visco" value="{{ nu }}" comment="Viscosity value. Typically 0.01 for Artificial and 1e-6 m^2/s (water kinematic viscosity) for Laminar" units_comment="m^2/s" />
            <parameter key="ViscoBoundFactor" value="1" comment="Multiply viscosity value with boundary (default=1)" />
            <parameter key="DensityDT" value="2" comment="Density Diffusion Term 0:None, 1:Molteni, 2:Fourtakas, 3:Fourtakas(full) (default=0)" />
            <parameter key="DensityDTvalue" value="0.1" comment="DDT value (default=0.1)" />
            <parameter key="Shifting" value="0" comment="Shifting mode 0:None, 1:Ignore bound, 2:Ignore fixed, 3:Full (default=0)" />
            <parameter key="RigidAlgorithm" value="1" comment="Rigid Algorithm 0:collision-free, 1:SPH, 2:DEM, 3:Chrono (default=1)" />
            <parameter key="FtPause" value="0.0" comment="Time to freeze the floatings at simulation start (warmup) (default=0)" units_comment="seconds" />
            <parameter key="CoefDtMin" value="0.05" comment="Coefficient to calculate minimum time step dtmin=coefdtmin*h/speedsound (default=0.05)" />
            <parameter key="DtIni" value="0.0001" comment="Initial time step. Use 0 to default use (default=h/speedsound)" units_comment="seconds" />
            <parameter key="DtMin" value="1e-8" comment="Minimum time step. Use 0 to default use (default=coefdtmin*h/speedsound)" units_comment="seconds" />
            <parameter key="DtFixed" value="0" comment="Fixed Dt value. Use 0 to disable (default=disabled)" units_comment="seconds" />
            <parameter key="TimeMax" value="{{ sim_time }}" comment="Time of simulation" units_comment="seconds" />
            <parameter key="TimeOut" value="{{ dt_out }}" comment="Time out data" units_comment="seconds" />
            <parameter key="RhopOutMin" value="700" comment="Minimum rhop valid (default=700)" units_comment="kg/m^3" />
            <parameter key="RhopOutMax" value="1300" comment="Maximum rhop valid (default=1300)" units_comment="kg/m^3" />
        </parameters>
        <output>
            <savevtk value="1" binary="0"/>
            <savecsv value="1"/>
            <csvpointinfo value="1"/>
            <measuretool value="1">
                <point id="ebay" x="{{ eb_x }}" y="{{ eb_y }}" z="{{ eb_z }}"/>
            </measuretool>
            <particletracking value="0"/>
        </output>
    </execution>
</case>
""")


# ── Drop impact case ─────────────────────────────────────────────────────
TEMPLATES["DROP_IMPACT_XML"] = DROP_IMPACT_XML = Template("""<?xml version="1.0" encoding="UTF-8"?>
<case>
    <casedef>
        <constantsdef>
            <lattice bound="2" fluid="1" comment="Type of lattice to create the initial particles (default=1)" />
            <gravity x="0" y="0" z="-{{ gravity }}" comment="Gravitational acceleration" units_comment="m/s^2" />
            <rhop0 value="{{ rho }}" comment="Reference density of the fluid" units_comment="kg/m^3" />
            <rhopgradient value="2" comment="Initial density gradient 1:Rhop0, 2:Water column, 3:Max. water height (default=2)" />
            <hswl value="0" auto="true" comment="Maximum still water level to calculate speedofsound using coefsound" units_comment="metres (m)" />
            <gamma value="7.0" comment="Polytropic constant for water used in the state equation" />
            <speedsystem value="0" auto="true" comment="Maximum system speed (by default the dam-break propagation is used)" />
            <coefsound value="20" comment="Coefficient to multiply speedsystem" />
            <speedsound value="0" auto="true" comment="Speed of sound to use in the simulation (by default speedofsound=coefsound*speedsystem)" />
            <hdp value="2" comment="Alternative option to calculate the smoothing length (h=hdp*dp)" />
            <cflnumber value="0.2" comment="Coefficient to multiply dt" />
        </constantsdef>
        <mkconfig boundcount="100" fluidcount="9" />
        <geometry>
            <definition dp="{{ dp }}" comment="Initial inter-particle distance" units_comment="metres (m)">
                <pointmin x="-{{ xmax + def_margin }}" y="-{{ ymax + def_margin }}" z="-{{ zmin + def_margin }}" />
                <pointmax x="{{ xmax + def_margin }}" y="{{ ymax + def_margin }}" z="{{ zmax + def_margin }}" />
            </definition>
            <commands>
                <mainlist>
                    <setshapemode>actual | bound</setshapemode>
                    <setdrawmode mode="full" />
                    <setmkbound mk="0" />
                    <drawbox cmt="Tank: fully closed box (all 6 walls)">
                        <boxfill>all</boxfill>
                        <point x="-{{ xmax }}" y="-{{ ymax }}" z="-{{ zmin }}" />
                        <size x="{{ 2*xmax }}" y="{{ 2*ymax }}" z="{{ zmax + zmin }}" />
                    </drawbox>
                    <setmkbound mk="1" />
                    <drawfilestl file="{{ hull_stl_path }}">
                        <drawmove x="{{ hull_x }}" y="0" z="{{ init_z }}" />
                    </drawfilestl>
                    <setmkfluid mk="0" />
                    <setboxlimitmode mode="full" />
                    <fillbox x="{{ seed_x }}" y="{{ seed_y }}" z="{{ seed_z }}">
                        <modefill>void</modefill>
                        <point x="-{{ xdomain }}" y="-{{ ydomain }}" z="-{{ zmin }}" />
                        <size x="{{ 2*xdomain }}" y="{{ 2*ydomain }}" z="{{ zsurf + zmin }}" />
                    </fillbox>
                    <shapeout file="" />
                </mainlist>
            </commands>
        </geometry>
        <floatings>
            <floating mkbound="1">
                <massbody value="{{ mass }}" />
                <inertia x="{{ Ixx }}" y="{{ Iyy }}" z="{{ Izz }}" />
                {{ cg_str }}
            </floating>
        </floatings>
        <initials>
            <densitydepth mkfluid="0" />
        </initials>
    </casedef>
    <execution>
        <special>
            <damping>
                <dampingzone>
                    <limitmin x="{{ x_damp }}" y="0" z="0" />
                    <limitmax x="{{ xmax }}" y="0" z="0" />
                    <overlimit value="1" />
                    <redumax value="10" />
                </dampingzone>
            </damping>
        </special>
        <constants>
            <data2d value="false"/>
            <gravity x="0" y="0" z="-{{ gravity }}"/>
            <cflnumber value="0.2"/>
            <gamma value="7.0"/>
            <rhop0 value="{{ rho }}"/>
            <eps value="0"/>
            <dp value="{{ dp }}"/>
            <h value="0"/>
            <b value="0"/>
            <massbound value="0"/>
            <massfluid value="0"/>
        </constants>
        <parameters>
            <parameter key="StepAlgorithm" value="2" comment="Step Algorithm 1:Verlet, 2:Symplectic (default=1)" />
            <parameter key="VerletSteps" value="50" comment="Verlet only: Number of steps to apply Euler timestepping (default=50)" />
            <parameter key="Kernel" value="2" comment="Interaction Kernel 1:Cubic Spline, 2:Wendland (default=2)" />
            <parameter key="ViscoTreatment" value="3" comment="Viscosity formulation 1:Artificial, 2:Laminar+SPS, 3:Laminar (default=1)" />
            <parameter key="Visco" value="{{ nu }}" comment="Viscosity value. Typically 0.01 for Artificial and 1e-6 m^2/s (water kinematic viscosity) for Laminar" units_comment="m^2/s" />
            <parameter key="ViscoBoundFactor" value="1" comment="Multiply viscosity value with boundary (default=1)" />
            <parameter key="DensityDT" value="2" comment="Density Diffusion Term 0:None, 1:Molteni, 2:Fourtakas, 3:Fourtakas(full) (default=0)" />
            <parameter key="DensityDTvalue" value="0.1" comment="DDT value (default=0.1)" />
            <parameter key="Shifting" value="0" comment="Shifting mode 0:None, 1:Ignore bound, 2:Ignore fixed, 3:Full (default=0)" />
            <parameter key="RigidAlgorithm" value="1" comment="Rigid Algorithm 0:collision-free, 1:SPH, 2:DEM, 3:Chrono (default=1)" />
            <parameter key="FtPause" value="0.0" comment="Time to freeze the floatings at simulation start (warmup) (default=0)" units_comment="seconds" />
            <parameter key="CoefDtMin" value="0.05" comment="Coefficient to calculate minimum time step dtmin=coefdtmin*h/speedsound (default=0.05)" />
            <parameter key="DtIni" value="0.0001" comment="Initial time step. Use 0 to default use (default=h/speedsound)" units_comment="seconds" />
            <parameter key="DtMin" value="1e-8" comment="Minimum time step. Use 0 to default use (default=coefdtmin*h/speedsound)" units_comment="seconds" />
            <parameter key="DtFixed" value="0" comment="Fixed Dt value. Use 0 to disable (default=disabled)" units_comment="seconds" />
            <parameter key="TimeMax" value="{{ sim_time }}" comment="Time of simulation" units_comment="seconds" />
            <parameter key="TimeOut" value="{{ dt_out }}" comment="Time out data" units_comment="seconds" />
            <parameter key="RhopOutMin" value="700" comment="Minimum rhop valid (default=700)" units_comment="kg/m^3" />
            <parameter key="RhopOutMax" value="1300" comment="Maximum rhop valid (default=1300)" units_comment="kg/m^3" />
        </parameters>
        <output>
            <savevtk value="1" binary="0"/>
            <savecsv value="1"/>
            <csvpointinfo value="1"/>
            <measuretool value="1">
                <point id="ebay" x="{{ eb_x }}" y="{{ eb_y }}" z="{{ eb_z }}"/>
            </measuretool>
            <particletracking value="0"/>
        </output>
    </execution>
</case>
""")


# ── Storm (irregular JONSWAP) case ───────────────────────────────────────
TEMPLATES["STORM_XML"] = STORM_XML = Template("""<?xml version="1.0" encoding="UTF-8"?>
<case>
    <casedef>
        <constantsdef>
            <lattice bound="2" fluid="1" comment="Type of lattice to create the initial particles (default=1)" />
            <gravity x="0" y="0" z="-{{ gravity }}" comment="Gravitational acceleration" units_comment="m/s^2" />
            <rhop0 value="{{ rho }}" comment="Reference density of the fluid" units_comment="kg/m^3" />
            <rhopgradient value="2" comment="Initial density gradient 1:Rhop0, 2:Water column, 3:Max. water height (default=2)" />
            <hswl value="0" auto="true" comment="Maximum still water level to calculate speedofsound using coefsound" units_comment="metres (m)" />
            <gamma value="7.0" comment="Polytropic constant for water used in the state equation" />
            <speedsystem value="0" auto="true" comment="Maximum system speed (by default the dam-break propagation is used)" />
            <coefsound value="20" comment="Coefficient to multiply speedsystem" />
            <speedsound value="0" auto="true" comment="Speed of sound to use in the simulation (by default speedofsound=coefsound*speedsystem)" />
            <hdp value="2" comment="Alternative option to calculate the smoothing length (h=hdp*dp)" />
            <cflnumber value="0.2" comment="Coefficient to multiply dt" />
        </constantsdef>
        <mkconfig boundcount="100" fluidcount="9" />
        <geometry>
            <definition dp="{{ dp }}" comment="Initial inter-particle distance" units_comment="metres (m)">
                <pointmin x="-{{ xmax + def_margin }}" y="-{{ ymax + def_margin }}" z="-{{ zmin + def_margin }}" />
                <pointmax x="{{ xmax + def_margin }}" y="{{ ymax + def_margin }}" z="{{ zmax + def_margin }}" />
            </definition>
            <commands>
                <mainlist>
                    <setshapemode>actual | bound</setshapemode>
                    <setdrawmode mode="full" />
                    <setmkbound mk="10" />
                    <drawbox cmt="Wave paddle (mk=10, driven by wavepaddles)">
                        <boxfill>solid</boxfill>
                        <point x="{{ x_paddle_lo }}" y="-{{ ymax }}" z="-{{ zmin }}" />
                        <size x="{{ x_paddle_th }}" y="{{ 2*ymax }}" z="{{ zmax + zmin }}" />
                    </drawbox>
                    <setmkbound mk="0" />
                    <drawbox cmt="Tank: fully closed box (all 6 walls)">
                        <boxfill>all</boxfill>
                        <point x="-{{ xmax }}" y="-{{ ymax }}" z="-{{ zmin }}" />
                        <size x="{{ 2*xmax }}" y="{{ 2*ymax }}" z="{{ zmax + zmin }}" />
                    </drawbox>
                    <setmkbound mk="1" />
                    <drawfilestl file="{{ hull_stl_path }}">
                        <drawmove x="{{ hull_x }}" y="0" z="{{ hull_z }}" />
                    </drawfilestl>
                    <setmkfluid mk="0" />
                    <setboxlimitmode mode="full" />
                    <fillbox x="{{ seed_x }}" y="{{ seed_y }}" z="{{ seed_z }}">
                        <modefill>void</modefill>
                        <point x="-{{ xdomain }}" y="-{{ ydomain }}" z="-{{ zmin }}" />
                        <size x="{{ 2*xdomain }}" y="{{ 2*ydomain }}" z="{{ still_water_level + zmin }}" />
                    </fillbox>
                    <shapeout file="" />
                </mainlist>
            </commands>
        </geometry>
        <motion>
            <objreal ref="10">
                <begin mov="1" start="0" />
                <mvnull id="1" />
            </objreal>
        </motion>
        <floatings>
            <floating mkbound="1">
                <massbody value="{{ mass }}" />
                <inertia x="{{ Ixx }}" y="{{ Iyy }}" z="{{ Izz }}" />
                {{ cg_str }}
            </floating>
        </floatings>
        <initials>
            <densitydepth mkfluid="0" />
        </initials>
    </casedef>
    <execution>
        <special>
            <wavepaddles>
                <piston_spectrum>
                    <mkbound value="10" />
                    <waveorder value="2" />
                    <start value="0" />
                    <duration value="0" />
                    <depth value="{{ water_depth }}" />
                    <pistondir x="1" y="0" z="0" />
                    <spectrum value="jonswap" />
                    <discretization value="stretched" />
                    <waveheight value="{{ Hs }}" />
                    <waveperiod value="{{ Tp }}" />
                    <peakcoef value="{{ gamma }}" />
                    <waves value="1024" />
                    <randomseed value="42" />
                    <serieini value="0" autofit="true" />
                    <ramptime value="{{ ramp_time }}" />
                </piston_spectrum>
            </wavepaddles>
            <damping>
                <dampingzone>
                    <limitmin x="{{ x_damp }}" y="0" z="0" />
                    <limitmax x="{{ xmax }}" y="0" z="0" />
                    <overlimit value="1" />
                    <redumax value="10" />
                </dampingzone>
            </damping>
        </special>
        <constants>
            <data2d value="false" />
            <gravity x="0" y="0" z="-{{ gravity }}" />
            <cflnumber value="0.2" />
            <gamma value="7.0" />
            <rhop0 value="{{ rho }}" />
            <eps value="0" />
            <dp value="{{ dp }}" />
            <h value="0" />
            <b value="0" />
            <massbound value="0" />
            <massfluid value="0" />
        </constants>
        <parameters>
            <parameter key="StepAlgorithm" value="2" comment="Step Algorithm 1:Verlet, 2:Symplectic (default=1)" />
            <parameter key="VerletSteps" value="50" comment="Verlet only: Number of steps to apply Euler timestepping (default=50)" />
            <parameter key="Kernel" value="2" comment="Interaction Kernel 1:Cubic Spline, 2:Wendland (default=2)" />
            <parameter key="ViscoTreatment" value="3" comment="Viscosity formulation 1:Artificial, 2:Laminar+SPS, 3:Laminar (default=1)" />
            <parameter key="Visco" value="{{ nu }}" comment="Viscosity value. Typically 0.01 for Artificial and 1e-6 m^2/s (water kinematic viscosity) for Laminar" units_comment="m^2/s" />
            <parameter key="ViscoBoundFactor" value="1" comment="Multiply viscosity value with boundary (default=1)" />
            <parameter key="DensityDT" value="2" comment="Density Diffusion Term 0:None, 1:Molteni, 2:Fourtakas, 3:Fourtakas(full) (default=0)" />
            <parameter key="DensityDTvalue" value="0.1" comment="DDT value (default=0.1)" />
            <parameter key="Shifting" value="0" comment="Shifting mode 0:None, 1:Ignore bound, 2:Ignore fixed, 3:Full (default=0)" />
            <parameter key="RigidAlgorithm" value="1" comment="Rigid Algorithm 0:collision-free, 1:SPH, 2:DEM, 3:Chrono (default=1)" />
            <parameter key="FtPause" value="0.0" comment="Time to freeze the floatings at simulation start (warmup) (default=0)" units_comment="seconds" />
            <parameter key="CoefDtMin" value="0.05" comment="Coefficient to calculate minimum time step dtmin=coefdtmin*h/speedsound (default=0.05)" />
            <parameter key="DtIni" value="0.0001" comment="Initial time step. Use 0 to default use (default=h/speedsound)" units_comment="seconds" />
            <parameter key="DtMin" value="1e-8" comment="Minimum time step. Use 0 to default use (default=coefdtmin*h/speedsound)" units_comment="seconds" />
            <parameter key="DtFixed" value="0" comment="Fixed Dt value. Use 0 to disable (default=disabled)" units_comment="seconds" />
            <parameter key="TimeMax" value="{{ sim_time }}" comment="Time of simulation" units_comment="seconds" />
            <parameter key="TimeOut" value="{{ dt_out }}" comment="Time out data" units_comment="seconds" />
            <parameter key="RhopOutMin" value="700" comment="Minimum rhop valid (default=700)" units_comment="kg/m^3" />
            <parameter key="RhopOutMax" value="1300" comment="Maximum rhop valid (default=1300)" units_comment="kg/m^3" />
        </parameters>
        <output>
            <savevtk value="1" binary="0"/>
            <savecsv value="1"/>
            <csvpointinfo value="1"/>
            <measuretool value="1"/>
            <particletracking value="0"/>
        </output>
    </execution>
</case>
""")


# ── Towing (calm-water resistance) case ──────────────────────────────────
# NOTE on XML dialect: the other templates in this file use the "new-style"
# v5.x XML (<parameters>/<geometry>/<shapes>/<motion type=...>). The GenCase
# binary installed at /home/anon/apps/DualSPHysics_v5.4 (v5.4.354.01)
# SEGFAULTS on that dialect (verified empirically) and only accepts the
# classic <casedef> format below (the one its own -templates output and
# every example in DualSPHysics_v5.4/examples use). This template therefore
# follows examples/inletoutlet/07_CurrentHull/CaseCurrentHull_Def.xml:
# a FIXED hull in a current driven by the native <inout> open-boundary
# mechanism (Tafuni et al. 2018), NOT a moving hull: the fluid is
# initialised at u0 (initials/velocity) and the inlet/outlet zones
# (execution/special/inout) maintain the current. Galilean-equivalent to a
# towed hull and cheaper than a moving-hull tank (no hull/domain-length
# constraint, matches the fixed-hull-at-speed interFoam setup). The hull
# cannot sink or pitch: it has no <floatings> block and no <motion>.
TEMPLATES["TOWING_XML"] = TOWING_XML = Template("""<?xml version="1.0" encoding="UTF-8"?>
<case>
    <casedef>
        <constantsdef>
            <lattice bound="2" fluid="1" comment="Type of lattice to create the initial particles (default=1)" />
            <gravity x="0" y="0" z="-{{ gravity }}" comment="Gravitational acceleration" units_comment="m/s^2" />
            <rhop0 value="{{ rho }}" comment="Reference density of the fluid" units_comment="kg/m^3" />
            <rhopgradient value="2" comment="Initial density gradient 1:Rhop0, 2:Water column, 3:Max. water height (default=2)" />
            <hswl value="0" auto="true" comment="Maximum still water level to calculate speedofsound using coefsound" units_comment="metres (m)" />
            <gamma value="7.0" comment="Polytropic constant for water used in the state equation" />
            <speedsystem value="{{ speedsystem }}" auto="false" comment="Maximum system speed (by default the dam-break propagation is used)" />
            <coefsound value="20" comment="Coefficient to multiply speedsystem" />
            <speedsound value="0" auto="true" comment="Speed of sound to use in the simulation (by default speedofsound=coefsound*speedsystem)" />
            <hdp value="2" comment="Alternative option to calculate the smoothing length (h=hdp*dp)" />
            <cflnumber value="0.2" comment="Coefficient to multiply dt" />
        </constantsdef>
        <mkconfig boundcount="100" fluidcount="9" />
        <geometry>
            <definition dp="{{ dp }}" comment="Initial inter-particle distance" units_comment="metres (m)">
                <pointmin x="{{ def_x0 }}" y="{{ def_y0 }}" z="{{ def_z0 }}" />
                <pointmax x="{{ def_x1 }}" y="{{ def_y1 }}" z="{{ def_z1 }}" />
            </definition>
            <commands>
                <mainlist>
                    <setshapemode>actual | bound</setshapemode>
                    <setdrawmode mode="full" />
                    <setmkbound mk="0" />
                    <drawbox cmt="Tow tank: closed box, open top (still water — the HULL moves, no imposed flow)">
                        <boxfill>bottom | front | back | left | right</boxfill>
                        <point x="{{ wall_x0 }}" y="{{ wall_y0 }}" z="{{ z_floor }}" />
                        <size x="{{ wall_dx }}" y="{{ wall_dy }}" z="{{ z_wall_top - z_floor }}" />
                    </drawbox>
                    <setmkbound mk="1" />
                    <drawfilestl file="{{ hull_stl_path }}">
                        <drawmove x="{{ hull_x }}" y="0" z="{{ hull_z }}" />
                    </drawfilestl>
                    <setmkfluid mk="0" />
                    <fillbox x="{{ seed_x }}" y="{{ seed_y }}" z="{{ seed_z }}">
                        <modefill>void</modefill>
                        <point x="{{ fluid_x0 }}" y="{{ fluid_y0 }}" z="{{ z_floor }}" />
                        <size x="{{ fluid_dx }}" y="{{ fluid_dy }}" z="{{ zsurf - z_floor }}" />
                    </fillbox>
                    <shapeout file="" />
                </mainlist>
            </commands>
        </geometry>
        <initials>
            <densitydepth mkfluid="0" />
        </initials>
        <motion>
            <objreal ref="{{ hull_ref }}">
                <begin mov="1" start="0.0" comment="constant-velocity carriage tow: hull moves through still water (Galilean-equivalent of towing at u0). No imposed flux — tank mass balance is trivially satisfied, so no level setup / Fz drift (fixed-hull flume variants both contaminated Fz, see Bug #157 follow-up 2026-08-25). NOTE: motion lives under casedef (GenCase reads it there), hull ref = creation index of the drawfilestl object." />
                <mvrect id="1" duration="{{ tow_duration }}">
                    <vel x="{{ u0 }}" y="0" z="0" units_comment="m/s" />
                </mvrect>
            </objreal>
        </motion>
    </casedef>
    <execution>
        <special>
            <damping cmt="Wave absorbers at both ends: bow waves radiate ahead of the hull, wake trails behind — neither may reflect off the end walls">
                <dampingbox>
                    <directions value="all,-top,-left,-right,-front,-back" comment="damp only in x" />
                    <limitmin comment="thin slab where minimum reduction starts (must lie inside limitmax)">
                        <pointini x="{{ damping_x0 }}" y="{{ ymin }}" z="{{ z_floor }}" />
                        <pointend x="{{ damping_x0 + 1.0 * dp }}" y="{{ ymax }}" z="{{ z_wall_top }}" />
                    </limitmin>
                    <limitmax comment="full damping region, max reduction from damping_x1 onward">
                        <pointini x="{{ damping_x0 }}" y="{{ ymin }}" z="{{ z_floor }}" />
                        <pointend x="{{ damping_x1 }}" y="{{ ymax }}" z="{{ z_wall_top }}" />
                    </limitmax>
                    <overlimit value="1" comment="keep max reduction beyond limitmax" />
                    <redumax value="10" />
                    <factorxyz x="1" y="1" z="1" />
                </dampingbox>
                <dampingbox>
                    <directions value="all,-top,-left,-right,-front,-back" comment="damp only in x" />
                    <limitmin comment="thin slab where minimum reduction starts (must lie inside limitmax)">
                        <pointini x="{{ damping2_x0 }}" y="{{ ymin }}" z="{{ z_floor }}" />
                        <pointend x="{{ damping2_x0 + 1.0 * dp }}" y="{{ ymax }}" z="{{ z_wall_top }}" />
                    </limitmin>
                    <limitmax comment="full damping region, max reduction from damping2_x1 onward">
                        <pointini x="{{ damping2_x0 }}" y="{{ ymin }}" z="{{ z_floor }}" />
                        <pointend x="{{ damping2_x1 }}" y="{{ ymax }}" z="{{ z_wall_top }}" />
                    </limitmax>
                    <overlimit value="1" comment="keep max reduction beyond limitmax" />
                    <redumax value="10" />
                    <factorxyz x="1" y="1" z="1" />
                </dampingbox>
            </damping>
        </special>
        <constants>
            <data2d value="false"/>
            <gravity x="0" y="0" z="-{{ gravity }}"/>
            <cflnumber value="0.2"/>
            <gamma value="7.0"/>
            <rhop0 value="{{ rho }}"/>
            <eps value="0"/>
            <dp value="{{ dp }}"/>
            <h value="0"/>
            <b value="0"/>
            <massbound value="0"/>
            <massfluid value="0"/>
        </constants>

        <parameters>
            <parameter key="StepAlgorithm" value="2" comment="Step Algorithm 1:Verlet, 2:Symplectic (default=1)" />
            <parameter key="VerletSteps" value="50" comment="Verlet only: Number of steps to apply Euler timestepping (default=50)" />
            <parameter key="Kernel" value="2" comment="Interaction Kernel 1:Cubic Spline, 2:Wendland (default=2)" />
            <parameter key="ViscoTreatment" value="3" comment="Viscosity formulation 1:Artificial, 2:Laminar+SPS, 3:Laminar (default=1)" />
            <parameter key="Visco" value="{{ nu }}" comment="Viscosity value. Typically 0.01 for Artificial and 1e-6 m^2/s (water kinematic viscosity) for Laminar" units_comment="m^2/s" />
            <parameter key="ViscoBoundFactor" value="1" comment="Multiply viscosity value with boundary (default=1)" />
            <parameter key="DensityDT" value="2" comment="Density Diffusion Term 0:None, 1:Molteni, 2:Fourtakas, 3:Fourtakas(full) (default=0)" />
            <parameter key="DensityDTvalue" value="0.1" comment="DDT value (default=0.1)" />
            <parameter key="Shifting" value="0" comment="Shifting mode 0:None, 1:Ignore bound, 2:Ignore fixed, 3:Full (default=0)" />
            <parameter key="RigidAlgorithm" value="1" comment="Rigid Algorithm 0:collision-free, 1:SPH, 2:DEM, 3:Chrono (default=1)" />
            <parameter key="FtPause" value="0.0" comment="Time to freeze the floatings at simulation start (warmup) (default=0)" units_comment="seconds" />
            <parameter key="CoefDtMin" value="0.05" comment="Coefficient to calculate minimum time step dtmin=coefdtmin*h/speedsound (default=0.05)" />
            <parameter key="DtIni" value="0.0001" comment="Initial time step. Use 0 to default use (default=h/speedsound)" units_comment="seconds" />
            <parameter key="DtMin" value="1e-8" comment="Minimum time step. Use 0 to default use (default=coefdtmin*h/speedsound)" units_comment="seconds" />
            <parameter key="DtFixed" value="0" comment="Fixed Dt value. Use 0 to disable (default=disabled)" units_comment="seconds" />
            <parameter key="TimeMax" value="{{ sim_time }}" comment="Time of simulation" units_comment="seconds" />
            <parameter key="TimeOut" value="{{ dt_out }}" comment="Time out data" units_comment="seconds" />
            <parameter key="RhopOutMin" value="700" comment="Minimum rhop valid (default=700)" units_comment="kg/m^3" />
            <parameter key="RhopOutMax" value="1300" comment="Maximum rhop valid (default=1300)" units_comment="kg/m^3" />
        </parameters>
    </execution>
</case>
""")


# ── Inverted hull (gate 5) case ──────────────────────────────────────────
# Calm water, no wavegen, no inout current. The hull is a FLOATING body
# starting INVERTED (deck down): the STL is pre-rotated 180° about the
# x-axis when written, so the keel points up out of the water and the deck
# (deepest point) is at the still-water level. The body sinks/settles under
# its own mass (floatings/massbody) and the peak pressure on the deck is
# captured with MeasureTool at post-processing.
TEMPLATES["INVERTED_XML"] = INVERTED_XML = Template("""<?xml version="1.0" encoding="UTF-8"?>
<case>
    <casedef>
        <constantsdef>
            <lattice bound="2" fluid="1" comment="Type of lattice to create the initial particles (default=1)" />
            <gravity x="0" y="0" z="-{{ gravity }}" comment="Gravitational acceleration" units_comment="m/s^2" />
            <rhop0 value="{{ rho }}" comment="Reference density of the fluid" units_comment="kg/m^3" />
            <rhopgradient value="2" comment="Initial density gradient 1:Rhop0, 2:Water column, 3:Max. water height (default=2)" />
            <hswl value="0" auto="true" comment="Maximum still water level to calculate speedofsound using coefsound" units_comment="metres (m)" />
            <gamma value="7.0" comment="Polytropic constant for water used in the state equation" />
            <speedsystem value="0" auto="true" comment="Maximum system speed (by default the dam-break propagation is used)" />
            <coefsound value="20" comment="Coefficient to multiply speedsystem" />
            <speedsound value="0" auto="true" comment="Speed of sound to use in the simulation (by default speedofsound=coefsound*speedsystem)" />
            <hdp value="2" comment="Alternative option to calculate the smoothing length (h=hdp*dp)" />
            <cflnumber value="0.2" comment="Coefficient to multiply dt" />
        </constantsdef>
        <mkconfig boundcount="100" fluidcount="9" />
        <geometry>
            <definition dp="{{ dp }}" comment="Initial inter-particle distance" units_comment="metres (m)">
                <pointmin x="{{ def_x0 }}" y="{{ def_y0 }}" z="{{ def_z0 }}" />
                <pointmax x="{{ def_x1 }}" y="{{ def_y1 }}" z="{{ def_z1 }}" />
            </definition>
            <commands>
                <mainlist>
                    <setshapemode>actual | bound</setshapemode>
                    <setdrawmode mode="full" />
                    <setmkbound mk="0" />
                    <drawbox cmt="Tank: fully closed box (bottom | front | back | left | right)">
                        <boxfill>bottom | front | back | left | right</boxfill>
                        <point x="{{ wall_x0 }}" y="{{ wall_y0 }}" z="{{ z_floor }}" />
                        <size x="{{ wall_dx }}" y="{{ wall_dy }}" z="{{ z_wall_top - z_floor }}" />
                    </drawbox>
                    <setmkbound mk="1" />
                    <drawfilestl file="{{ hull_stl_path }}">
                        <drawmove x="{{ hull_x }}" y="0" z="{{ hull_z }}" />
                    </drawfilestl>
                    <setmkfluid mk="0" />
                    <setboxlimitmode mode="full" />
                    <fillbox x="{{ seed_x }}" y="{{ seed_y }}" z="{{ seed_z }}">
                        <modefill>void</modefill>
                        <point x="{{ fluid_x0 }}" y="{{ fluid_y0 }}" z="{{ z_floor }}" />
                        <size x="{{ fluid_dx }}" y="{{ fluid_dy }}" z="{{ zsurf - z_floor }}" />
                    </fillbox>
                    <shapeout file="" />
                </mainlist>
            </commands>
        </geometry>
        <floatings>
            <floating mkbound="1">
                <massbody value="{{ mass }}" />
                <inertia x="{{ Ixx }}" y="{{ Iyy }}" z="{{ Izz }}" />
                {{ cg_str }}
            </floating>
        </floatings>
        <initials>
            <densitydepth mkfluid="0" />
        </initials>
    </casedef>
    <execution>
        <constants>
            <data2d value="false"/>
            <gravity x="0" y="0" z="-{{ gravity }}"/>
            <cflnumber value="0.2"/>
            <gamma value="7.0"/>
            <rhop0 value="{{ rho }}"/>
            <eps value="0"/>
            <dp value="{{ dp }}"/>
            <h value="0"/>
            <b value="0"/>
            <massbound value="0"/>
            <massfluid value="0"/>
        </constants>

        <parameters>
            <parameter key="StepAlgorithm" value="2" comment="Step Algorithm 1:Verlet, 2:Symplectic (default=1)" />
            <parameter key="VerletSteps" value="50" comment="Verlet only: Number of steps to apply Euler timestepping (default=50)" />
            <parameter key="Kernel" value="2" comment="Interaction Kernel 1:Cubic Spline, 2:Wendland (default=2)" />
            <parameter key="ViscoTreatment" value="3" comment="Viscosity formulation 1:Artificial, 2:Laminar+SPS, 3:Laminar (default=1)" />
            <parameter key="Visco" value="{{ nu }}" comment="Viscosity value. Typically 0.01 for Artificial and 1e-6 m^2/s (water kinematic viscosity) for Laminar" units_comment="m^2/s" />
            <parameter key="ViscoBoundFactor" value="1" comment="Multiply viscosity value with boundary (default=1)" />
            <parameter key="DensityDT" value="2" comment="Density Diffusion Term 0:None, 1:Molteni, 2:Fourtakas, 3:Fourtakas(full) (default=0)" />
            <parameter key="DensityDTvalue" value="0.1" comment="DDT value (default=0.1)" />
            <parameter key="Shifting" value="0" comment="Shifting mode 0:None, 1:Ignore bound, 2:Ignore fixed, 3:Full (default=0)" />
            <parameter key="RigidAlgorithm" value="1" comment="Rigid Algorithm 0:collision-free, 1:SPH, 2:DEM, 3:Chrono (default=1)" />
            <parameter key="FtPause" value="0.0" comment="Time to freeze the floatings at simulation start (warmup) (default=0)" units_comment="seconds" />
            <parameter key="CoefDtMin" value="0.05" comment="Coefficient to calculate minimum time step dtmin=coefdtmin*h/speedsound (default=0.05)" />
            <parameter key="DtIni" value="0.0001" comment="Initial time step. Use 0 to default use (default=h/speedsound)" units_comment="seconds" />
            <parameter key="DtMin" value="1e-8" comment="Minimum time step. Use 0 to default use (default=coefdtmin*h/speedsound)" units_comment="seconds" />
            <parameter key="DtFixed" value="0" comment="Fixed Dt value. Use 0 to disable (default=disabled)" units_comment="seconds" />
            <parameter key="TimeMax" value="{{ sim_time }}" comment="Time of simulation" units_comment="seconds" />
            <parameter key="TimeOut" value="{{ dt_out }}" comment="Time out data" units_comment="seconds" />
            <parameter key="RhopOutMin" value="700" comment="Minimum rhop valid (default=700)" units_comment="kg/m^3" />
            <parameter key="RhopOutMax" value="1300" comment="Maximum rhop valid (default=1300)" units_comment="kg/m^3" />
        </parameters>
    </execution>
</case>
""")

def _sph_stl_path(hull_stl_path: str) -> str:
    """Resolve the STL for the SPH cases.

    Prefers the hull+keel+bulb combined mesh (``hull_full.stl``, written
    alongside ``hull.stl`` by generate_hull) so the SPH cases see the full
    underwater geometry; falls back to the hull-only STL for older runs
    without the combined export.
    """
    p = Path(hull_stl_path)
    full = p.parent / "hull_full.stl"
    if full.exists():
        return str(full.resolve())
    return str(p.resolve())


def _stl_z_bounds(stl_path: str) -> Optional[tuple[float, float]]:
    """(z_min, z_max) of the STL in mesh frame, or None if unreadable."""
    try:
        import trimesh
        mesh = trimesh.load(stl_path, force="mesh")
        return float(mesh.bounds[0, 2]), float(mesh.bounds[1, 2])
    except Exception:
        return None


def storm_domain(LWL: float, B: float, T: float, Hs: float, dp: float) -> dict:
    """Storm-tank geometry for a hull of length LWL, beam B, draft T.

    Single source of truth shared by write_storm_case (case generation) and
    sph_gates._estimate_storm_dp (particle-count budgeting): any change to
    the tank sizing automatically keeps the estimator in sync.
    """
    xmax = 1.5 * LWL
    ymax = 1.5 * B
    zmin = T * 3.0
    still_water_level = max(T + 0.1, 5.0 * dp)
    # Domain top must clear the still water level (with wave crest headroom):
    # a fillbox that protrudes beyond the definition box yields zero fluid.
    zmax = max(Hs * 2.5 + 0.5, still_water_level + 0.5)
    xdomain = xmax
    ydomain = ymax
    zdomain = max(zmax, zmin)
    # Floor the domain/water level so the fillbox always has positive height
    # (>= 5 particle layers) — zero-height fluid crashes the solver with
    # "Constant 'b' cannot be zero".
    zdomain = max(zdomain, 5.0 * dp)
    tank_depth = zdomain
    # Real water column height (tank floor -> still water level): the piston
    # "depth" parameter is the fluid depth, not the floor depth below z=0.
    water_depth = zmin + still_water_level
    # Irregular-wave paddle stroke scales ~ wave height, and the 2nd-order
    # piston transfer approaches ~1 in shallow transfer regimes: peak travel
    # over a 1024-wave JONSWAP realization reaches ~Hs. The old fixed 0.35 m
    # pocket let the paddle exit the particle-derived domain on the return
    # stroke (AbortBoundOut, -X face) for shallow narrow hulls
    # (ref_ds_41/91/121/141/161 — all B=0.55). Size the pocket from Hs
    # (Bug #167); the particle estimator stays in sync via storm_domain.
    # Paddle stays >= 1.5*dp thick so GenCase builds a solid boundary wall.
    stroke_est = Hs
    pocket = max(0.35, stroke_est + 4.0 * dp)
    x_paddle_lo = -xmax + pocket
    x_paddle_th = max(0.10, 2.0 * dp)
    # Wave-damping slab ahead of the far-end wall kills the wave energy before
    # it reflects off the closed end.  Start well clear of the hull (hull
    # spans [-LWL/2, +LWL/2]) and of the piston (waves need to develop).
    x_damp = 0.7 * xmax
    return {
        "xmax": xmax, "ymax": ymax, "zmin": zmin,
        "still_water_level": still_water_level,
        "zmax": zmax,
        "xdomain": xdomain, "ydomain": ydomain, "zdomain": zdomain,
        "tank_depth": tank_depth, "water_depth": water_depth,
        "x_paddle_lo": x_paddle_lo, "x_paddle_th": x_paddle_th,
        "x_damp": x_damp,
    }


def write_storm_case(case_dir: Path, hull_stl_path: str,
                     LWL: float, B: float, T: float, mass: float,
                     Hs: float = 0.25, Tp: float = 1.3,
                     gamma: float = 3.3,
                     gravity: float = 9.81, rho: float = 1025.0,
                     nu: float = 1.0e-6,
                     sim_time: float = 10.0,
                     dt_out: float = 0.1,
                     dp: float = 0.02,
                     eb_coords: tuple = (0.0, 0.0, -0.05),
                     cg_z: Optional[float] = None,
                     cg_x: Optional[float] = None):
    case_dir = Path(case_dir)
    case_dir.mkdir(parents=True, exist_ok=True)

    Ixx = mass * (B ** 2 + T ** 2) / 12.0
    Iyy = mass * (LWL ** 2 + T ** 2) / 12.0
    Izz = mass * (LWL ** 2 + B ** 2) / 12.0

    if cg_x is None:
        cg_x = LWL * 0.4
    if cg_z is not None:
        cg_str = f'<center x="{cg_x}" y="0" z="{cg_z}"/>'
    else:
        cg_str = f'<center x="{cg_x}" y="0" z="{-T * 0.4}"/>'

    dom = storm_domain(LWL, B, T, Hs, dp)
    xmax = dom["xmax"]
    ymax = dom["ymax"]
    zmin = dom["zmin"]
    still_water_level = dom["still_water_level"]
    zmax = dom["zmax"]
    xdomain = dom["xdomain"]
    ydomain = dom["ydomain"]
    zdomain = dom["zdomain"]
    tank_depth = dom["tank_depth"]
    water_depth = dom["water_depth"]
    x_paddle_lo = dom["x_paddle_lo"]
    x_paddle_th = dom["x_paddle_th"]
    ramp_time = 0.5 * Tp
    x_damp = dom["x_damp"]

    # hull surface measure point at side of hull at waterline
    hs_x = LWL * 0.7
    hs_y = B * 0.5
    hs_z = 0.0

    # fillbox seed inside the fluid box, clear of the bounding tank walls and
    # in front of the wave paddle.
    seed_x = x_paddle_lo + x_paddle_th + 5.0 * dp
    seed_y = -ydomain + 5.0 * dp
    seed_z = -zmin + 5.0 * dp

    # Float the hull at its design draft: the deepest STL point (keel or bulb
    # tip) rests at still_water_level - T.  Fallback assumes the full-mesh STL
    # (deepest point at -T), which keeps the hull at draft even if the mesh
    # cannot be read.
    hull_stl_path = _sph_stl_path(hull_stl_path)
    z_bounds = _stl_z_bounds(hull_stl_path)
    if z_bounds is not None:
        hull_z = (still_water_level - T) - z_bounds[0]
    else:
        hull_z = still_water_level
    hull_x = -LWL / 2.0

    content = STORM_XML.render(
        gravity=gravity, rho=rho, nu=nu,
        sim_time=sim_time, dt_out=dt_out,
        dp=dp,
        xmax=xmax, ymax=ymax, zmin=zmin, zmax=zmax,
        def_margin=DEF_MARGIN,
        xdomain=xdomain, ydomain=ydomain, zdomain=zdomain,
        x_paddle_lo=x_paddle_lo, x_paddle_th=x_paddle_th,
        x_damp=x_damp, ramp_time=ramp_time,
        seed_x=seed_x, seed_y=seed_y, seed_z=seed_z,
        hull_stl_path=hull_stl_path,
        hull_x=hull_x, hull_z=hull_z,
        mass=mass, Ixx=Ixx, Iyy=Iyy, Izz=Izz,
        cg_str=cg_str,
        tank_depth=tank_depth,
        water_depth=water_depth,
        still_water_level=still_water_level,
        Hs=Hs, Tp=Tp, gamma=gamma,
    )

    xml_path = case_dir / "case_storm.xml"
    xml_path.write_text(content)

    # Write measure points for post-processing
    points_path = case_dir / "measure_points.txt"
    lines = ["POINTS"]
    lines.append(f"{eb_coords[0]:.6f} {eb_coords[1]:.6f} {eb_coords[2]:.6f}")
    lines.append(f"{hs_x:.6f} {hs_y:.6f} {hs_z:.6f}")
    points_path.write_text("\n".join(lines) + "\n")

    return xml_path


def write_focused_wave_case(case_dir: Path, hull_stl_path: str,
                            LWL: float, B: float, T: float, mass: float,
                            gravity: float = 9.81, rho: float = 1025.0,
                            nu: float = 1.0e-6,
                            wave_height: float = 2.4,
                            wave_period: float = 5.0,
                            sim_time: float = 15.0,
                            dt_out: float = 0.05,
                            dp: float = 0.02,
                            eb_coords: tuple = (0.0, 0.0, -0.05),
                            cg_z: Optional[float] = None,
                            cg_x: Optional[float] = None):
    case_dir = Path(case_dir)
    case_dir.mkdir(parents=True, exist_ok=True)

    Ixx = mass * (B ** 2 + T ** 2) / 12.0
    Iyy = mass * (LWL ** 2 + T ** 2) / 12.0
    Izz = mass * (LWL ** 2 + B ** 2) / 12.0

    if cg_x is None:
        cg_x = LWL * 0.4
    if cg_z is not None:
        cg_str = f'<center x="{cg_x}" y="0" z="{cg_z}"/>'
    else:
        cg_str = f'<center x="{cg_x}" y="0" z="{-T * 0.4}"/>'

    xmax = 1.5 * LWL
    ymax = 1.5 * B
    zmin = T * 3.0
    still_water_level = max(T + 0.1, 5.0 * dp)
    # Domain top must clear the still water level: a fillbox that protrudes
    # beyond the definition box yields zero fluid.
    zmax = max(wave_height * 1.5, still_water_level + 0.5)
    xdomain = xmax
    ydomain = ymax
    zdomain = max(zmax, zmin)
    # Floor the domain/water level so the fillbox always has positive height
    # (>= 5 particle layers) — zero-height fluid crashes the solver with
    # "Constant 'b' cannot be zero".
    zdomain = max(zdomain, 5.0 * dp)
    # Real water column height (tank floor -> still water level): the piston
    # "depth" parameter is the fluid depth, not the floor depth below z=0.
    water_depth = zmin + still_water_level
    # Paddle geometry: rest position 0.35 m inside the -X end (back pocket
    # clears the piston back-stroke), thickness >= 1.5*dp so GenCase builds a
    # solid moving wall.
    x_paddle_lo = -xmax + 0.35
    x_paddle_th = max(0.10, 2.0 * dp)
    ramp_time = 0.5 * wave_period
    # Wave-damping slab ahead of the closed far end.
    x_damp = 0.7 * xmax
    # Focus the wave crest at the hull centreline.
    xf = 0.0

    # fillbox seed inside the fluid box, clear of the bounding tank walls and
    # in front of the wave paddle.
    seed_x = x_paddle_lo + x_paddle_th + 5.0 * dp
    seed_y = -ydomain + 5.0 * dp
    seed_z = -zmin + 5.0 * dp

    # Float the hull at its design draft: the deepest STL point (keel or bulb
    # tip) rests at still_water_level - T.  Fallback assumes the full-mesh STL
    # (deepest point at -T), which keeps the hull at draft even if the mesh
    # cannot be read.
    hull_stl_path = _sph_stl_path(hull_stl_path)
    z_bounds = _stl_z_bounds(hull_stl_path)
    if z_bounds is not None:
        hull_z = (still_water_level - T) - z_bounds[0]
    else:
        hull_z = still_water_level
    hull_x = -LWL / 2.0

    content = FOCUSED_WAVE_XML.render(
        gravity=gravity, rho=rho, nu=nu,
        sim_time=sim_time, dt_out=dt_out,
        dp=dp,
        xmax=xmax, ymax=ymax, zmin=zmin, zmax=zmax,
        def_margin=DEF_MARGIN,
        xdomain=xdomain, ydomain=ydomain, zdomain=zdomain,
        x_paddle_lo=x_paddle_lo, x_paddle_th=x_paddle_th,
        x_damp=x_damp, ramp_time=ramp_time, xf=xf,
        seed_x=seed_x, seed_y=seed_y, seed_z=seed_z,
        hull_stl_path=hull_stl_path,
        hull_x=hull_x, hull_z=hull_z,
        mass=mass, Ixx=Ixx, Iyy=Iyy, Izz=Izz,
        cg_str=cg_str,
        water_depth=water_depth,
        still_water_level=still_water_level,
        wave_height=wave_height, wave_period=wave_period,
        eb_x=eb_coords[0], eb_y=eb_coords[1], eb_z=eb_coords[2],
    )

    xml_path = case_dir / "case_focused_wave.xml"
    xml_path.write_text(content)


def write_drop_impact_case(case_dir: Path, hull_stl_path: str,
                           LWL: float, B: float, T: float, mass: float,
                           drop_height: float = 1.5,
                           gravity: float = 9.81, rho: float = 1025.0,
                           nu: float = 1.0e-6,
                           sim_time: float = 3.0,
                           dt_out: float = 0.001,
                           dp: float = 0.02,
                           eb_coords: tuple = (0.0, 0.0, -0.05),
                           cg_z: Optional[float] = None):
    case_dir = Path(case_dir)
    case_dir.mkdir(parents=True, exist_ok=True)

    Ixx = mass * (B ** 2 + T ** 2) / 12.0
    Iyy = mass * (LWL ** 2 + T ** 2) / 12.0
    Izz = mass * (LWL ** 2 + B ** 2) / 12.0
    if cg_z is None:
        cg_z = -T * 0.4
    cg_str = f'<center x="{LWL * 0.4}" y="0" z="{cg_z}"/>'

    xmax = 2.0 * LWL
    ymax = 2.0 * B
    zmax = drop_height + T + 0.5
    zmin = T * 3.0
    xdomain = xmax
    ydomain = ymax

    # initial velocity at water impact from drop height
    import numpy as np
    init_vel_z = -np.sqrt(2.0 * gravity * drop_height)

    # fillbox seed inside the fluid box, clear of the bounding tank walls.
    # The fillbox spans the domain's water column: [-zmin, zsurf].
    seed_x = -xdomain + 5.0 * dp
    seed_y = -ydomain + 5.0 * dp
    seed_z = -zmin + 5.0 * dp

    # Drop the hull so its deepest point (keel/bulb tip) starts drop_height
    # above the water surface, and make sure the domain top clears the hull
    # when held at that height.  Fallback assumes the full-mesh STL.
    hull_stl_path = _sph_stl_path(hull_stl_path)
    z_bounds = _stl_z_bounds(hull_stl_path)
    if z_bounds is not None:
        z_stl_min, z_stl_max = z_bounds
    else:
        z_stl_min, z_stl_max = -T, 0.0
    init_z = drop_height - z_stl_min
    zmax = max(zmax, init_z + z_stl_max + 0.5)
    zdomain = max(zmax, zmin)
    # Floor the domain so the fillbox always has positive height
    # (>= 5 particle layers) — zero-height fluid crashes the solver with
    # "Constant 'b' cannot be zero".
    zdomain = max(zdomain, 5.0 * dp)
    # Wave-damping slab ahead of the closed far end: kills the impact ripples
    # before they reflect off the +X wall back into the hull.
    x_damp = 0.7 * xmax

    content = DROP_IMPACT_XML.render(
        gravity=gravity, rho=rho, nu=nu,
        sim_time=sim_time, dt_out=dt_out,
        dp=dp,
        xmax=xmax, ymax=ymax, zmin=zmin, zmax=zmax,
        def_margin=DEF_MARGIN,
        xdomain=xdomain, ydomain=ydomain, zdomain=zdomain,
        x_damp=x_damp,
        seed_x=seed_x, seed_y=seed_y, seed_z=seed_z,
        hull_stl_path=hull_stl_path,
        mass=mass, Ixx=Ixx, Iyy=Iyy, Izz=Izz,
        cg_str=cg_str,
        init_z=init_z,
        hull_x=-LWL / 2.0,
        zsurf=0.0,
        eb_x=eb_coords[0], eb_y=eb_coords[1], eb_z=eb_coords[2],
    )

    xml_path = case_dir / "case_drop_impact.xml"
    xml_path.write_text(content)


def write_towing_case(case_dir: Path, hull_stl_path: str,
                      LWL: float, B: float, T_canoe: float, D_keel: float,
                      speed_ms: float, dp: float,
                      sim_time: float = 15.0,
                      dt_out: float = 0.02,
                      mass: float = 150.0,
                      rho: float = 1025.0,
                      nu: float = 1.0e-6,
                      gravity: float = 9.81,
                      eb_coords: tuple = (0.0, 0.0, -0.05),
                       keel_chord: float = 0.0):  # noqa: unused, kept for API compat
    """Towing test v3: the HULL moves through still water at constant speed.

    Root-cause fix (Bug #157 follow-up, 2026-08-25): both fixed-hull flume
    variants contaminated the vertical force — an imposed-velocity outlet
    traps the wave system (radiation pressure, Fz drift +405 N/s), while an
    extrapolated outlet lets the forced inlet flux pile water up until the
    level rises (Fz -> 11.6 kN vs 2.9 kN sanity bound). A moving hull in
    still water has no global mass balance to violate: Fz stays bounded and
    the measured Fx is a true carriage-tow drag.

    Geometry: closed tank (open top), hull starts near the left end and is
    towed +x by an mvrect constant-velocity motion; damping zones at both
    ends absorb bow/wake waves. The requested sim_time is capped so the
    travel fits the particle budget (~48 m^3 fluid at dp -> ~385k particles).
    """
    import numpy as np

    case_dir = Path(case_dir)
    case_dir.mkdir(parents=True, exist_ok=True)
    hull_stl_path = _sph_stl_path(hull_stl_path)

    zsurf = 0.0
    T_total = T_canoe + D_keel
    # Floor: keel-tip clearance of 1.5 keel chords (ground effect is mild at
    # gap/chord >= 1.5), never shallower than 0.4 m and never deeper than
    # 3.2 m (particle budget).
    z_floor = -(T_total + max(1.5 * abs(keel_chord), 0.4))
    z_floor = max(z_floor, -3.2)
    # Ensure the fillbox always has positive height (>= 5 particle layers).
    z_floor = min(z_floor, zsurf - max(5.0 * dp, 0.01))
    # The definition domain must contain the whole hull (a deck at E=0.45
    # pokes above the old zsurf+0.15 ceiling and gets clipped by GenCase).
    z_wall_top = zsurf + 0.15
    stl_bounds = _stl_z_bounds(hull_stl_path)
    if stl_bounds is not None:
        z_wall_top = max(z_wall_top, zsurf + stl_bounds[1] + 1.0 * dp)

    u0 = speed_ms
    speedsystem = max(u0 * 1.5, 2.0)

    # Tank cross-section: width 2.0*B (wall effects on the moving-hull wave
    # field are second-order for drag over the short window), depth from floor.
    ymax = max(1.0 * B, 0.55)
    ymin = -ymax
    depth = z_wall_top - z_floor

    # Travel fits the particle budget: V = tank_len * W * D <= 44 m^3
    # (~380k total particles incl. walls at dp=0.05).
    # tank_len = front + LWL(hull) + travel + back with front = 0.6*LWL,
    # back = 0.5*LWL (damping + clearance; the stern must stay inside the
    # tank or AbortBoundOut kills the run — observed 2026-08-25 when the
    # margin bookkeeping omitted the hull length).
    budget_v = 44.0
    margin_total = 2.1 * LWL
    max_len = budget_v / max(2.0 * ymax * depth, 1e-6)
    travel_max = max(0.8 * LWL, max_len - margin_total)
    sim_eff = min(float(sim_time), travel_max / max(u0, 1e-6))
    if sim_eff < float(sim_time):
        logger.warning(
            "Towing sim_time capped %.2f -> %.2f s to fit the particle "
            "budget (travel %.2f m, tank %.2f m)", sim_time, sim_eff,
            u0 * sim_eff, travel_max + margin_total,
        )
    tank_len = u0 * sim_eff + margin_total

    x0 = 0.0
    x1 = tank_len
    hull_x = x0 + 0.6 * LWL          # bow (STL x=0) starting position
    tow_duration = sim_eff + 1.0     # hull never decelerates inside the run

    # Damping zones: ahead of the bow start (radiated bow waves) and around
    # the final stern position (wake).
    damping_x0 = x0 + 0.05
    damping_x1 = x0 + 0.5 * LWL
    damping2_x0 = x1 - 1.1 * LWL
    damping2_x1 = x1 - 0.1

    fluid_x0 = x0 - 2.0 * dp
    fluid_dx = tank_len + 4.0 * dp

    # fillbox seed near mid-tank, clear of the bounding tank walls
    seed_x = fluid_x0 + 0.5 * fluid_dx
    seed_y = ymin + 5.0 * dp
    seed_z = z_floor + 5.0 * dp

    content = TOWING_XML.render(
        gravity=gravity, rho=rho, nu=nu,
        sim_time=sim_eff, dt_out=dt_out,
        dp=dp,
        def_x0=x0 - DEF_MARGIN, def_y0=ymin - DEF_MARGIN, def_z0=z_floor - DEF_MARGIN,
        def_x1=x1 + DEF_MARGIN, def_y1=ymax + DEF_MARGIN, def_z1=z_wall_top + DEF_MARGIN,
        wall_x0=x0 - 2.0 * dp, wall_y0=ymin - 2.0 * dp,
        wall_dx=tank_len + 4.0 * dp, wall_dy=2.0 * ymax + 4.0 * dp,
        fluid_x0=fluid_x0, fluid_y0=ymin, fluid_dx=fluid_dx, fluid_dy=2.0 * ymax,
        seed_x=seed_x, seed_y=seed_y, seed_z=seed_z,
        zsurf=zsurf, z_floor=z_floor,
        z_wall_top=z_wall_top,
        hull_stl_path=hull_stl_path,
        hull_x=hull_x, hull_z=0.0,
        hull_ref=1,  # objreal ref = geometry creation index (0=tank box, 1=hull STL); NOT the mkbound mk
        u0=u0, speedsystem=speedsystem,
        tow_duration=tow_duration,
        ymin=ymin, ymax=ymax,
        damping_x0=damping_x0, damping_x1=damping_x1,
        damping2_x0=damping2_x0, damping2_x1=damping2_x1,
    )

    xml_path = case_dir / "case_towing.xml"
    xml_path.write_text(content)

    # measure point at the electronics bay, in tank frame (hull frame offset
    # by the bow start position)
    eb_tank = (eb_coords[0] + hull_x, eb_coords[1], eb_coords[2] - T_canoe)
    points_path = case_dir / "measure_points.txt"
    lines = ["POINTS"]
    lines.append(f"{eb_tank[0]:.6f} {eb_tank[1]:.6f} {eb_tank[2]:.6f}")
    points_path.write_text("\n".join(lines) + "\n")

    return xml_path


def write_inverted_case(case_dir: Path, hull_stl_path: str,
                        LWL: float, B: float, T_canoe: float, D_keel: float,
                        mass: float,
                        dp: float,
                        sim_time: float = 15.0,
                        dt_out: float = 0.02,
                        rho: float = 1025.0,
                        nu: float = 1.0e-6,
                        gravity: float = 9.81,
                        cg_z: Optional[float] = None,
                        eb_coords: tuple = (0.0, 0.0, -0.05),
                        keel_chord: float = 0.0,  # noqa: unused, kept for API compat
                        inertia: Optional[tuple] = None,
                        max_particles: int = 400000):
    """Inverted self-righting test: hull capsized (deck down, keel up).

    The STL is rotated 180 degrees about the x axis and placed with the deck
    at the still water surface (floating flat). The hull then sinks to its
    upside-down equilibrium draft (found by volume bisection on the rotated
    mesh). Measure points are written at the equilibrium deck depth plus a
    small clearance (2.5 dp) so they register the hydrostatic/dynamic head on
    the deck as the vessel settles and starts to roll back.
    """
    import trimesh
    import numpy as np

    case_dir = Path(case_dir)
    case_dir.mkdir(parents=True, exist_ok=True)
    hull_stl_path = _sph_stl_path(hull_stl_path)

    mesh = trimesh.load(hull_stl_path, force="mesh")
    rot = trimesh.transformations.rotation_matrix(
        np.pi, [1.0, 0.0, 0.0], point=[0.0, 0.0, 0.0]
    )
    mesh.apply_transform(rot)

    inverted_path = case_dir / "inverted_hull.stl"
    mesh.export(str(inverted_path))

    z0, z1 = float(mesh.bounds[0, 2]), float(mesh.bounds[1, 2])
    z_sheer_max = z1  # post-flip keel-tip-up height (measured, not param)
    T_total = T_canoe + D_keel
    # Measured draft beats param draft: bulb overlap/rake shift the true tip.
    T_meas = max(z1, T_total)
    if inertia is not None:
        Ixx, Iyy, Izz = (float(inertia[0]), float(inertia[1]), float(inertia[2]))
    else:
        Ixx = mass * (B ** 2 + T_total ** 2) / 12.0
        Iyy = mass * (LWL ** 2 + T_total ** 2) / 12.0
        Izz = mass * (LWL ** 2 + B ** 2) / 12.0
    if cg_z is None:
        cg_z = -T_total * 0.4
    zsurf = 0.0

    # upside-down equilibrium draft FIRST (domain derives from it below):
    # with drawmove z = dz, the rotated hull sits at tank_z = z' + dz, so
    # the submerged part (tank_z < 0) is the region z' < -dz. Find dz so
    # that its volume displaces mass/rho of water.
    target_vol = mass / rho
    lo = -3.0 * T_canoe - z1
    hi = z1

    def _submerged_vol(dz):
        cut = trimesh.intersections.slice_mesh_plane(
            mesh, plane_origin=[0.0, 0.0, -dz], plane_normal=[0.0, 0.0, -1.0],
            cap=True,
        )
        return float(cut.volume) if cut is not None else 0.0

    if _submerged_vol(lo) < target_vol:
        # cannot float upside down: sinks to the floor
        dz_eq = lo
    else:
        for _ in range(40):
            mid = 0.5 * (lo + hi)
            if _submerged_vol(mid) > target_vol:
                lo = mid
            else:
                hi = mid
        dz_eq = 0.5 * (lo + hi)

    # Tank sized to the MEASURED motion envelope (Bug #166). The solver
    # domain (MapRealPos) is the INITIAL particle bbox + (KernelH*0.05 +
    # dp/2) — the def box is NOT the domain. Tip/deck/CG positions below
    # are all in the tank frame (post-rotation, post-drawmove).
    z_tip = dz_eq + z1
    z_dk = dz_eq + z0
    z_cg = -cg_z + dz_eq
    R = max(abs(z_tip - z_cg), abs(z_dk - z_cg), 0.5 * B)

    x0 = -1.5 * LWL
    x1 = +1.5 * LWL
    hull_x = -LWL / 2.0

    def _tank(dp_):
        _h = 2.0 * dp_  # smoothing-length scale
        _marg = _h * 0.05 + dp_ / 2.0 + 2.0 * dp_  # kernel + lattice + bob
        _wtop = z_tip + 0.5 + _marg
        _floor = min(-1.5 * T_meas, z_dk - 0.3) - _marg
        # fillbox must keep positive height (>= 5 layers) or GenCase makes
        # zero fluid ("Constant 'b' cannot be zero").
        _floor = min(_floor, zsurf - max(5.0 * dp_, 0.01))
        # ymax covers the self-righting roll swing (radius R about the CG).
        _ymax = R + 0.3 + _marg
        return _wtop, _floor, _ymax, _marg

    # Adaptive dp (Bug #162 pattern): fluid volume grows ~T^2, so deep-keel
    # hulls blow the particle cap at fixed dp. Coarsen before GenCase.
    z_wall_top, z_floor, ymax, marg = _tank(dp)
    _vfluid = (x1 - x0) * (2.0 * ymax) * (zsurf - z_floor)
    while _vfluid / max(dp ** 3, 1e-12) > max_particles and dp < 0.12:
        dp *= 1.2
        z_wall_top, z_floor, ymax, marg = _tank(dp)
        _vfluid = (x1 - x0) * (2.0 * ymax) * (zsurf - z_floor)
    ymin = -ymax
    logger.info(f"inverted case: T_meas={T_meas:.2f} dz_eq={dz_eq:.2f} "
                f"walls=[{z_floor:.2f},{z_wall_top:.2f}] ymax={ymax:.2f} dp={dp:.4f}")

    wall_box = (
        (x0 - 2.0 * dp, ymin - 2.0 * dp, z_floor - 2.0 * dp),
        (x1 - x0 + 4.0 * dp, 2.0 * ymax + 4.0 * dp, z_wall_top - z_floor + 2.0 * dp),
    )
    fillbox = (
        (x0, ymin, z_floor),
        (x1 - x0, 2.0 * ymax, zsurf - z_floor),
    )

    # upside-down equilibrium draft was solved above (dz_eq) before the
    # domain was derived from it, so the walls already cover tip +/- bob.

    # deck surface depth at equilibrium, from ray casts on the rotated mesh:
    # a ray up from below hits the deck (lowest point of the inverted hull)
    # first; tank_z of the deck at equilibrium is z' + dz_eq (negative).
    ray_origins = []
    for xf in (0.15, 0.3, 0.5, 0.7, 0.85):
        for yf in (-0.3, 0.0, 0.3):
            ray_origins.append((xf * LWL - LWL / 2.0, yf * B, -1.0))
    ray_origins = np.array(ray_origins)
    dirs = np.zeros_like(ray_origins)
    dirs[:, 2] = 1.0
    pts, index_ray, _ = mesh.ray.intersects_location(ray_origins, dirs)
    deck_z = {}
    for ray_i in range(len(ray_origins)):
        hit = pts[index_ray == ray_i]
        if len(hit) == 0:
            continue
        z_hit = float(np.min(hit[:, 2]))
        key = (round(float(ray_origins[ray_i, 0]), 3),
               round(float(ray_origins[ray_i, 1]), 3))
        z_tank = z_hit + dz_eq
        if key not in deck_z or z_tank < deck_z[key]:
            deck_z[key] = z_tank
    if not deck_z:
        deck_z[(-LWL / 2.0, 0.0)] = -z_sheer_max / 2.0

    # The mesh is rotated 180 deg about x (keel now points UP); the floating
    # body <center> must be given in the TANK frame (post-rotation, post-
    # drawmove): z_tank = -cg_z + dz_eq, x_tank = hull_x + LWL*0.4. The old
    # code passed the upright-frame cg_z (=-0.60) directly and forgot the
    # hull_x offset, leaving the center ~1.2 m from the true mass center ->
    # spurious torque -> the body spun out and AbortBoundOut at t~0.14 s.
    cg_str = f'<center x="{hull_x + LWL * 0.4}" y="0" z="{-cg_z + dz_eq}"/>'

    # fillbox seed inside the fluid box, clear of the bounding tank walls
    seed_x = x0 + 5.0 * dp
    seed_y = ymin + 5.0 * dp
    seed_z = z_floor + 5.0 * dp

    content = INVERTED_XML.render(
        gravity=gravity, rho=rho, nu=nu,
        sim_time=sim_time, dt_out=dt_out,
        dp=dp,
        def_x0=x0 - DEF_MARGIN, def_y0=ymin - DEF_MARGIN, def_z0=z_floor - DEF_MARGIN,
        def_x1=x1 + DEF_MARGIN, def_y1=ymax + DEF_MARGIN,
        # def box covers the measured tip + bob headroom (walls already do);
        # it must exceed the walls or GenCase drops flush faces (DEF_MARGIN).
        def_z1=z_wall_top + DEF_MARGIN,
        wall_x0=x0 - 2.0 * dp, wall_y0=ymin - 2.0 * dp,
        wall_dx=x1 - x0 + 4.0 * dp, wall_dy=2.0 * ymax + 4.0 * dp,
        fluid_x0=x0, fluid_y0=ymin, fluid_dx=x1 - x0, fluid_dy=2.0 * ymax,
        seed_x=seed_x, seed_y=seed_y, seed_z=seed_z,
        zsurf=zsurf, z_floor=z_floor,
        z_wall_top=z_wall_top,
        hull_stl_path=str(inverted_path),
        hull_x=hull_x, hull_z=dz_eq,
        mass=mass,
        Ixx=Ixx, Iyy=Iyy, Izz=Izz,
        cg_str=cg_str,
    )

    xml_path = case_dir / "case_inverted.xml"
    xml_path.write_text(content)

    # measure points at the equilibrium deck depth + clearance
    points_path = case_dir / "measure_points.txt"
    lines = ["POINTS"]
    for (x_tank, y_tank), z_deck in deck_z.items():
        lines.append(f"{x_tank:.6f} {y_tank:.6f} {z_deck + 2.5 * dp:.6f}")
    # electronics bay point in tank frame: rotated (z -> -z) then offset by
    # the equilibrium draft
    eb_tank = (eb_coords[0] - LWL / 2.0, -eb_coords[1], -eb_coords[2] + dz_eq)
    lines.append(f"{eb_tank[0]:.6f} {eb_tank[1]:.6f} {eb_tank[2]:.6f}")
    points_path.write_text("\n".join(lines) + "\n")

    return xml_path
