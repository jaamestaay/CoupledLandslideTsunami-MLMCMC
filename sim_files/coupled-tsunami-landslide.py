import peano4
import exahype2
import math 
import os 
import sys 

rusanov_fv = r"""
  tests::exahype2::CoupledLandslideTsunami_MLMCMC::sim_files::LandslideTsunamiSolver s;
  s.flux(QL, x, h, t, dt, normal, FL);
  s.flux(QR, x, h, t, dt, normal, FR);
  const auto smax{std::fmax(s.maxEigenvalue(QL, x, h, t, dt, normal), s.maxEigenvalue(QR, x, h, t, dt, normal))};
  const auto N{NumberOfAuxiliaryVariables + NumberOfUnknowns};
  double Qavg[N] = {};
  double DeltaQ[N] = {};
  for (int i = 0; i < NumberOfUnknowns; ++i) {
    Qavg[i]   = 0.5 * (QL[i] + QR[i]);
    DeltaQ[i] = QR[i] - QL[i];
    FL[i]     = 0.5 * (FL[i] + FR[i] - smax * DeltaQ[i]);
    FR[i]     = FL[i];
  }
  for (int i = NumberOfUnknowns; i < N; ++i) {
    Qavg[i]   = 0.5 * (QL[i] + QR[i]);
    DeltaQ[i] = QR[i] - QL[i];
  }
  double ncp[NumberOfUnknowns] = {};
  s.nonconservativeProduct(Qavg, DeltaQ, x, h, t, dt, normal, ncp);
  for (int i = 0; i < NumberOfUnknowns; ++i) {
    FL[i] += 0.5 * ncp[i];
    FR[i] -= 0.5 * ncp[i];
  }
  auto sL{0.0};
  if (QL[s::h1] > hThreshold) {
    const auto hL{QL[s::h1] > hThreshold ? QL[s::h1] : hThreshold};
    const auto momentumL{std::sqrt(QL[s::h1u] * QL[s::h1u] + QL[s::h1v] * QL[s::h1v])};
    sL = 2.0 * g * momentumL * invXi / (hL * hL);
  }
  auto sR{0.0};
  if (QL[s::h1] > hThreshold) {
    const auto hR{QR[s::h1] > hThreshold ? QR[s::h1] : hThreshold};
    const auto momentumR{std::sqrt(QR[s::h1u] * QR[s::h1u] + QR[s::h1v] * QR[s::h1v])};
    sR = 2.0 * g * momentumR * invXi / (hR * hR);
  }
  return std::fmax(smax, h[normal] * std::fmax(sL, sR));
"""

stiff_source_term_fv = r"""
  for (int n = 0; n < NumberOfUnknowns; n++) {
    S[n] = 0.0;
  }

  if (Q[s::h1] <= hThreshold) {
    return;
  }

  // Compute local slope cF
  const auto dzx{deltaQ[s::z]};
  const auto dzy{deltaQ[s::z + NumberOfUnknowns + NumberOfAuxiliaryVariables]};
  const auto cF{1.0 / std::sqrt(1.0 + dzx * dzx + dzy * dzy)};

  // Apply friction only if there is some movement
  const auto momentumSQR{Q[s::h1u] * Q[s::h1u] + Q[s::h1v] * Q[s::h1v]};
  const auto momentum{std::sqrt(momentumSQR)};
  if (tarch::la::greater(momentum / Q[s::h1], 0.0)) {
    const auto coulombFriction{mu * cF * Q[s::h1]};
    const auto frictionTerm{-g * coulombFriction / momentum};
    S[s::h1u] = Q[s::h1u] * frictionTerm;
    S[s::h1v] = Q[s::h1v] * frictionTerm;
  }
"""

project = exahype2.Project(
    namespace=["tests", "exahype2", "CoupledLandslideTsunami_MLMCMC", "sim_files"],
    project_name="CoupledLandslideTsunamis",
    directory=".",
    executable="CoupledLandslideTsunamis"
)

project.set_output_path("solutions")

parser = exahype2.ArgumentParser()
parser.add_argument(
    "--friction",
    type=float,
    help="Friction parameter.",
) # might have to add more arguments to incorporate into the system
parser.set_defaults(
    min_depth=4,
    end_time=.5,
    degrees_of_freedom=7,
    time_step_relaxation=0.45,
    friction=1/200.0
)
args = parser.parse_args()

end_time = 5.0
dimensions = 2
size = [60.0, 60.0]
dg_order = args.degrees_of_freedom - 1
max_h = (1.1 * min(size) / (3.0**args.min_depth))
min_h = max_h * 3.0 ** (-args.amr_levels)



constants = {
    "g": [9.81, "double"],
    "phi": [25.0, "double"],
    # "invXi": [args.friction, "double"],
    "hThreshold": [1e-1, "double"],
}
constants["mu"] = [
    math.tan(math.pi / 180.0 * constants["phi"][0]),
    "double",
]
with open("params.cpp", "w") as f:
  f.write("constexpr double invXi = " + str(args.friction) + ";")

project.set_global_simulation_parameters(
    dimensions=dimensions,
    size=size,
    offset=[0.0, 0.0],
    min_end_time=end_time,
    max_end_time=end_time,
    first_plot_time_stamp=0.0,
    time_in_between_plots=.1,
    periodic_BC=[False, False]
)


### DEFINE ALL VARIABLES FOR COUPLED MODEL
unknowns = {"h0": 1, "h0u": 1, "h0v": 1, "b": 1, "h1": 1, "h1u": 1, "h1v": 1, "z": 1}
auxiliary_variables = {}


my_solver = exahype2.solvers.fv.godunov.GlobalAdaptiveTimeStep(
    name                  = "LandslideTsunamiSolver",
    min_volume_h          = min_h,
    max_volume_h          = max_h,
    patch_size            = dg_order * 2 + 1,
    unknowns              = unknowns,
    auxiliary_variables   = auxiliary_variables,
    time_step_relaxation  = args.time_step_relaxation,
)

# my_solver = exahype2.solvers.fv.godunov.GlobalFixedTimeStep(
#     name                      = "LandslideTsunamiSolver",
#     min_volume_h              = min_h,
#     max_volume_h              = max_h,
#     patch_size                = dg_order * 2 + 1,
#     unknowns                  = unknowns,
#     auxiliary_variables       = auxiliary_variables,
#     normalised_time_step_size = args.time_step_relaxation,
# )

my_solver.set_implementation(
    flux=exahype2.solvers.PDETerms.User_Defined_Implementation,
    ncp=exahype2.solvers.PDETerms.User_Defined_Implementation,
    max_eigenvalue=exahype2.solvers.PDETerms.User_Defined_Implementation,
    boundary_conditions=exahype2.solvers.PDETerms.User_Defined_Implementation,
    initial_conditions=exahype2.solvers.PDETerms.User_Defined_Implementation,
    diffusive_source_term=stiff_source_term_fv,
    riemann_solver=rusanov_fv # what is used when not specified? Some differnce in the SWE calculations. 
)

my_solver.set_plotter(args.plotter)

my_solver.plot_description = ", ".join(unknowns.keys())
project.add_solver(my_solver)

project.set_load_balancer(f"new ::exahype2::LoadBalancingConfiguration({args.load_balancing_quality}, 1, {args.trees_init}, {args.trees})")
project.set_Peano4_installation("../../Peano", mode=peano4.output.string_to_mode(args.build_mode))
project = project.generate_Peano4_project(verbose=False)
for const_name, const_info in constants.items():
    const_val, const_type = const_info
    project.constants.export_constexpr_with_type(
        const_name, str(const_val), const_type
    )
project.output.makefile.set_target_device(args.target_device)
project.set_fenv_handler(args.fpe)
project.build(make=True, make_clean_first=True, throw_away_data_after_build=True)