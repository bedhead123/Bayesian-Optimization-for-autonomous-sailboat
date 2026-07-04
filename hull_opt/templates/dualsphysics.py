from pathlib import Path
from typing import Optional
from jinja2 import Template


# ── Focused wave group case ──────────────────────────────────────────────
FOCUSED_WAVE_XML = Template("""<?xml version="1.0" encoding="UTF-8"?>
<case>
  <parameters>
    <gravity value="0 0 -{{ gravity }}"/>
    <rhop0 value="{{ rho }}"/>
    <gamma value="7.0"/>
    <visco value="{{ nu }}"/>
    <tf value="{{ sim_time }}"/>
    <dtinit value="0.0001"/>
    <dtmin value="1e-8"/>
    <dtout value="{{ dt_out }}"/>
    <cflnumber value="0.2"/>
    <svp value="1.0"/>
    <dt_ini value="0.1"/>
  </parameters>

  <geometry>
    <shapes>
      <shape type="box" value="{{ -xmax }} {{ -ymax }} {{ -zmin }} {{ xmax }} {{ ymax }} {{ zmax }}">
        <motion type="fixed"/>
      </shape>
      <shape type="triangulation" file="{{ hull_stl_path }}">
        <motion type="floating">
          <mass value="{{ mass }}"/>
          <inertia Ixx="{{ Ixx }}" Iyy="{{ Iyy }}" Izz="{{ Izz }}"/>
          {{ cg_str }}
          <dof type="all"/>
          <floatinginfo>
            <limandler type="dtlimit" value="0.8"/>
            <velmax type="all" value="1.0"/>
            <massini value="true"/>
          </floatinginfo>
        </motion>
      </shape>
    </shapes>
  </geometry>

  <fluid>
    <filling box="-{{ xdomain }} -{{ ydomain }} -{{ zdomain }} {{ xdomain }} {{ ydomain }} {{ zdomain }}" value="1"/>
    <reserve block="1.5"/>
  </fluid>

  <wavegen>
    <pistonwave>
      <depth value="{{ tank_depth }}"/>
      <hswl value="{{ still_water_level }}"/>
      <wavetype type="focused">
        <height value="{{ wave_height }}"/>
        <period value="{{ wave_period }}"/>
        <timefocus value="{{ time_focus }}"/>
        <Nwaves value="1"/>
      </wavetype>
      <piston type="piston"/>
    </pistonwave>
  </wavegen>

  <execution>
    <gpu value="1"/>
    <dp value="1"/>
    <noregions value="0"/>
    <verletsteps value="50"/>
  </execution>

  <output>
    <savevtk value="1" binary="0"/>
    <savecsv value="1"/>
    <csvpointinfo value="1"/>
    <measuretool value="1">
      <point id="ebay" x="{{ eb_x }}" y="{{ eb_y }}" z="{{ eb_z }}"/>
    </measuretool>
    <particletracking value="0"/>
  </output>
</case>
""")


# ── Drop impact case ─────────────────────────────────────────────────────
DROP_IMPACT_XML = Template("""<?xml version="1.0" encoding="UTF-8"?>
<case>
  <parameters>
    <gravity value="0 0 -{{ gravity }}"/>
    <rhop0 value="{{ rho }}"/>
    <gamma value="7.0"/>
    <visco value="{{ nu }}"/>
    <tf value="{{ sim_time }}"/>
    <dtinit value="0.00001"/>
    <dtmin value="1e-9"/>
    <dtout value="{{ dt_out }}"/>
    <cflnumber value="0.1"/>
    <svp value="1.0"/>
    <dt_ini value="0.01"/>
  </parameters>

  <geometry>
    <shapes>
      <shape type="box" value="{{ -xmax }} {{ -ymax }} {{ -zmin }} {{ xmax }} {{ ymax }} {{ zmax }}">
        <motion type="fixed"/>
      </shape>
      <shape type="triangulation" file="{{ hull_stl_path }}">
        <motion type="prescribed">
          <initial pos="{{ init_x }} {{ init_y }} {{ init_z }}" vel="0 0 {{ init_vel_z }}" angular="0 0 0"/>
          <falling type="freefall" stop_time="10.0"/>
          <dof type="all"/>
        </motion>
      </shape>
    </shapes>
  </geometry>

  <fluid>
    <filling box="-{{ xdomain }} -{{ ydomain }} -{{ zdomain }} {{ xdomain }} {{ ydomain }} {{ zdomain }}" value="1"/>
    <reserve block="1.5"/>
  </fluid>

  <execution>
    <gpu value="1"/>
    <dp value="1"/>
    <noregions value="0"/>
    <verletsteps value="50"/>
  </execution>

  <output>
    <savevtk value="1" binary="0"/>
    <savecsv value="1"/>
    <csvpointinfo value="1"/>
    <measuretool value="1">
      <point id="ebay" x="{{ eb_x }}" y="{{ eb_y }}" z="{{ eb_z }}"/>
    </measuretool>
    <particletracking value="0"/>
  </output>
</case>
""")


def write_focused_wave_case(case_dir: Path, hull_stl_path: str,
                            LWL: float, B: float, T: float, mass: float,
                            gravity: float = 9.81, rho: float = 1025.0,
                            nu: float = 1.0e-6,
                            wave_height: float = 2.4,
                            wave_period: float = 5.0,
                            sim_time: float = 15.0,
                            dt_out: float = 0.05,
                            eb_coords: tuple = (0.0, 0.0, -0.05),
                            cg_z: Optional[float] = None):
    case_dir = Path(case_dir)
    case_dir.mkdir(parents=True, exist_ok=True)

    Ixx = mass * (B ** 2 + T ** 2) / 12.0
    Iyy = mass * (LWL ** 2 + T ** 2) / 12.0
    Izz = mass * (LWL ** 2 + B ** 2) / 12.0

    if cg_z is not None:
        cg_str = f'<cg x="0" y="0" z="{cg_z}"/>'
    else:
        cg_str = f'<cg x="0" y="0" z="{-T * 0.4}"/>'

    xmax = 1.5 * LWL
    ymax = 1.5 * B
    zmax = wave_height * 1.5
    zmin = T * 3.0
    xdomain = xmax
    ydomain = ymax
    zdomain = max(zmax, zmin)
    tank_depth = zdomain
    still_water_level = T + 0.1
    time_focus = sim_time * 0.4

    content = FOCUSED_WAVE_XML.render(
        gravity=gravity, rho=rho, nu=nu,
        sim_time=sim_time, dt_out=dt_out,
        xmax=xmax, ymax=ymax, zmin=zmin, zmax=zmax,
        xdomain=xdomain, ydomain=ydomain, zdomain=zdomain,
        hull_stl_path=hull_stl_path,
        mass=mass, Ixx=Ixx, Iyy=Iyy, Izz=Izz,
        cg_str=cg_str,
        tank_depth=tank_depth,
        still_water_level=still_water_level,
        wave_height=wave_height, wave_period=wave_period,
        time_focus=time_focus,
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
                           eb_coords: tuple = (0.0, 0.0, -0.05)):
    case_dir = Path(case_dir)
    case_dir.mkdir(parents=True, exist_ok=True)

    xmax = 2.0 * LWL
    ymax = 2.0 * B
    zmax = drop_height + T + 0.5
    zmin = T * 3.0
    xdomain = xmax
    ydomain = ymax
    zdomain = max(zmax, zmin)

    # initial velocity at water impact from drop height
    import numpy as np
    init_vel_z = -np.sqrt(2.0 * gravity * drop_height)

    content = DROP_IMPACT_XML.render(
        gravity=gravity, rho=rho, nu=nu,
        sim_time=sim_time, dt_out=dt_out,
        xmax=xmax, ymax=ymax, zmin=zmin, zmax=zmax,
        xdomain=xdomain, ydomain=ydomain, zdomain=zdomain,
        hull_stl_path=hull_stl_path,
        init_x=0.0, init_y=0.0,
        init_z=drop_height,
        init_vel_z=init_vel_z,
        eb_x=eb_coords[0], eb_y=eb_coords[1], eb_z=eb_coords[2],
    )

    xml_path = case_dir / "case_drop_impact.xml"
    xml_path.write_text(content)
