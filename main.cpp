#include <fstream>
#include <iostream>
#include <dolfin.h>
#include <stdlib.h>
#include <cmath>
#include <time.h>
#include <vector>
#include <stack>

// Parameters used for computational domain
#define nx  30
#define ny  30
#define nz  30

// Parameter for domain size
#define xmin -0.75
#define xmax  1.125
#define ymin -0.75
#define ymax  0.75
#define zmin -0.60
#define zmax  1.125

#define tinc_ 1e-3

// Define material parameter and stabilization parameter
#define rhoa_         1.27
#define rhow_         1.0e3
#define mua_          2.0e-5
#define muw_          1.0e-3

#define water_inlet_  1.5
#define air_inlet_    0.1
#define zref_         0.3

#define epslen_       2.0e-2
#define epsset_       1.0e-10

#define lc_kdc_       0.0

#define pse_dt_       2.0e-2
#define phid_kdc_     1.0
#define phid_penalty_ 1.0e4

// Frequently used parameters here
#define tpi 3.141592653589793
#define distol 1e-4

using namespace dolfin;

#include "ns.h"
#include "rd.h"
#include "octree.h"
#include "redis.h"

double cal_Heaviside( double phi ){
	double res;

	if( phi < -epslen_){
		res = 0.0;
	}
	else if( phi > epslen_){
		res = 1.0;
	}
	else{
		res = 0.5*( 1.0 + phi/epslen_ + 1.0/tpi*std::sin( phi*tpi/epslen_ ) );
	}

	return res;
};

// Define Inlet  domain
class InletDomain : public SubDomain
{
	bool inside(const Array<double>& x, bool on_boundary) const
	{
		return (on_boundary and (x[0] < xmin + distol) );
	}
};

// Define Outlet domain
class OutletDomain : public SubDomain
{
	bool inside(const Array<double>& x, bool on_boundary) const
	{
		return (on_boundary and (x[0] > xmax - distol) );
	}
};

// Define Y direction domain
class YDomain : public SubDomain
{
	bool inside(const Array<double>& x, bool on_boundary) const
	{
		return (on_boundary and ( (x[1] > ymax - distol) or (x[1] < ymin + distol) ) );
	}
};

// Define Z direction domain
class ZDomain : public SubDomain
{
	bool inside(const Array<double>& x, bool on_boundary) const
	{
		return (on_boundary and ( (x[2] > zmax - distol) or (x[2] < zmin + distol) ) );
	}
};

class AllDomain : public SubDomain
{
	bool inside(const Array<double>& x, bool on_boundary) const
	{
		return true ;
	}
};

class CylDomain : public SubDomain
{
    bool inside(const Array<double>& x, bool on_boundary) const
    {
        return (on_boundary and (std::abs( x[0] ) < 0.05) and (std::abs( x[1] ) < 0.05) and (x[2] < 0.6) and (x[2] > -0.1) );
    }
};

// Define center domain
class PreDomain : public SubDomain
{
	bool inside(const Array<double>& x, bool on_boundary) const
	{
		double tol = 1e-4;
		return ( std::abs( x[0] - xmin )<tol and std::abs( x[1] - ymin )<tol and std::abs( x[2] - zmin )<tol );
	}
};

// Initial conditions defined here:
class InitialConditions : public Expression
{
	public:

	//====Becareful the rank of variable is 0===============//
	//====Set Expression(1) would make rank inconsistent====//
	InitialConditions() : Expression(5)
	{
		dolfin::seed(2 + dolfin::MPI::rank(MPI_COMM_WORLD));
	}

	void eval(Array<double>& values, const Array<double>& x) const
	{
		double dist = zref_ - x[2];
		double Hval = cal_Heaviside(dist);
		
		values[0] = water_inlet_*Hval + air_inlet_*(1.0 - Hval);
		values[1]= 0.0;
		values[2]= 0.0;
		values[3]= 0.0;

		values[4]= dist;
	}

};

// User defined nonlinear problem
class EquationSol : public NonlinearProblem
{
	public:

	// Constructor
	EquationSol(std::shared_ptr<const Form> F,
			std::shared_ptr<const Form> J,
			std::vector<DirichletBC*> bcs) : _F(F), _J(J), _bcs(bcs) {}

	// User defined residual vector
	void F(GenericVector& b, const GenericVector& x)
	{
		assemble(b, *_F);
		for (std::size_t i = 0; i < _bcs.size(); i++)
			_bcs[i]->apply(b, x);
	}

	// User defined assemble of Jacobian
	void J(GenericMatrix& A, const GenericVector& x)
	{
		assemble(A, *_J);
		for (std::size_t i = 0; i < _bcs.size(); i++)
			_bcs[i]->apply(A);
	}

	void bcschange(std::vector<DirichletBC*> bcnew)
	{
		for (std::size_t i = 0; i < _bcs.size(); i++)
			_bcs[i]=bcnew[i];
	}

	private:

	// Forms
	std::shared_ptr<const Form> _F;
	std::shared_ptr<const Form> _J;
	std::vector<DirichletBC*> _bcs;
};

// body force
class Source : public dolfin::Expression
{
	public:
	
	Source( )
	: Expression(3) {}

	void eval(Array<double>& values, const Array<double>& x) const
	{
		values[0]=  0.0;
		values[1]=  0.0;
		values[2]= -9.8;
	}

};

class InletVel : public dolfin::Expression
{
	public:
	
	InletVel( )
	: Expression(3) {}

	void eval(Array<double>& values, const Array<double>& x) const
	{
		double dist = zref_ - x[2];
		double Hval = cal_Heaviside(dist);
		
		values[0] = water_inlet_*Hval + air_inlet_*(1.0 - Hval);
		values[1]= 0.0;
		values[2]= 0.0;
	}

};

// Mesh velocity
class Meshvel : public dolfin::Expression
{
	public:
	
	Meshvel(double Omega)
	: Expression(3), _Omega(Omega) {}

	void eval(Array<double>& values, const Array<double>& x) const
	{
		values[0]= 0.0;
		values[1]= 0.0;
		values[2]= 0.0;
	}

	private:
		double _Omega;
};

class Phitop : public dolfin::Expression
{
	public:

	Phitop(double t)
		: Expression( ), _t(t) {}

	void eval(Array<double>& values, const Array<double>& x) const
	{
		values[0]= zref_ - x[2];
	}

	public:
		double _t;
};

void read_step(int& nstep, int& tnum, double& t)
{
	int inum;
	std::string step_name ="step.dat";
	std::ifstream stepFile;
	stepFile.open(step_name, std::ios::in);
	stepFile >> inum;
	stepFile.close();

	nstep = inum;

	if(nstep == 0){
		tnum = 0;
	}
	else{
		tnum = nstep+1;
	}

	t = tnum*tinc_;
}

int main(int argc, char* argv[])
{
	init(argc, argv);

	PetscErrorCode ierr;

	// Backend
	parameters["linear_algebra_backend"] = "PETSc";

	int rankvalues=dolfin::MPI::rank(MPI_COMM_WORLD);
	bool ismaster = ( dolfin::MPI::rank(MPI_COMM_WORLD) == 0 );

	double time;
	int nstep,tnum;
	if(ismaster) info("Read step");
	read_step(nstep,tnum,time);

	/*
	// Used for BoxMesh:
	Point a0(x0, 	y0, 	z0);
	Point a1(x0+lx, y0+ly, 	z0+lz);

	auto min = BoxMesh::create(MPI_COMM_WORLD,{a0, a1}, {nx,ny,nz},CellType::Type::tetrahedron);
	auto mesh= std::make_shared<Mesh>(min);
	*/

	auto mesh = std::make_shared<Mesh>("cyl.xml");

	/*
	auto mesh = std::make_shared<Mesh>(MPI_COMM_WORLD);
	auto f1 = HDF5File(MPI_COMM_WORLD,"sp-coarse.h5", "r");
	f1.read(*mesh, "mesh", false);
	*/

	// Define domain
	auto inlet_domain = std::make_shared<InletDomain>();
	auto out_domain   = std::make_shared<OutletDomain>();
	auto y_domain     = std::make_shared<YDomain>();
	auto z_domain     = std::make_shared<ZDomain>();
	auto cen_domain   = std::make_shared<PreDomain>();
	auto all_domain   = std::make_shared<AllDomain>();
    auto cyl_domain   = std::make_shared<CylDomain>();

	// Define the marker of top domain:
	auto sub_domains = std::make_shared<MeshFunction<std::size_t>>(mesh, mesh->topology().dim() - 1);
	*sub_domains = 0;
	inlet_domain->mark(*sub_domains, 1);
	out_domain->mark(*sub_domains,   2);

	//auto periodic_boundary = std::make_shared<PeriodicBoundary>();
	//auto W = std::make_shared<ns::FunctionSpace>(mesh,periodic_boundary);
	auto W = std::make_shared<ns::FunctionSpace>(mesh);
	auto F = std::make_shared<ns::LinearForm>(W);
	auto J = std::make_shared<ns::JacobianForm>(W, W);

	auto Wrd = std::make_shared<rd::FunctionSpace>(mesh);
	//auto Frd = std::make_shared<rd::LinearForm>(W);
	//auto Jrd = std::make_shared<rd::JacobianForm>(W, W);

	std::vector<std::size_t> d2v_map      = dof_to_vertex_map(*Wrd);
	std::vector<dolfin::la_index> v2d_map = vertex_to_dof_map(*Wrd);

	// Passing marker to Bilinear form and Functional here:
	J->ds = sub_domains;
	F->ds = sub_domains;

	// Define variable for boundary
	auto zero          = std::make_shared<Constant>(0.0);
	auto zero_vector   = std::make_shared<Constant>(0.0, 0.0, 0.0);
	auto inlet_vector  = std::make_shared<InletVel>();
	auto ur            = std::make_shared<Meshvel> (0.0);

	DirichletBC vel_inlet    (W->sub(0)                 , inlet_vector         , inlet_domain );
	DirichletBC acc_inlet    (W->sub(0)                 , zero_vector          , inlet_domain );
	
	DirichletBC vel_ydir     (W->sub(0)->sub(1)         , zero                 , y_domain     );
	DirichletBC acc_ydir     (W->sub(0)->sub(1)         , zero                 , y_domain     );

	DirichletBC vel_zdir     (W->sub(0)->sub(2)         , zero                 , z_domain     );
	DirichletBC acc_zdir     (W->sub(0)->sub(2)         , zero                 , z_domain     );

    DirichletBC vel_cyl      (W->sub(0)                 , zero_vector          , cyl_domain   );
    DirichletBC acc_cyl      (W->sub(0)                 , zero_vector          , cyl_domain   );

	DirichletBC pre_cen      (W->sub(1)                 , zero                 , cen_domain,  "pointwise");
	//DirichletBC pre_cen      (W->sub(1)                 , zero                 , out_domain   );
	
	DirichletBC dphi_inlet   (W->sub(2)                 , zero                 , inlet_domain );

	// Boundary condition for u,phi
	std::vector<DirichletBC*> bcs_ori   = {{&vel_inlet, &vel_ydir, &vel_zdir, &vel_cyl  }};
	// Boundary condition for du,p,dphi
	std::vector<DirichletBC*> bcs_der   = {{&acc_inlet, &acc_ydir, &acc_zdir, &pre_cen, &acc_cyl  }};
	// Boundary condition for Newton solver
	std::vector<DirichletBC*> bcs       = {{&acc_inlet, &acc_ydir, &acc_zdir, &pre_cen, &acc_cyl  }};
	// Boundary condition for rd solver
	std::vector<DirichletBC*> bcs_rd    = {};

	auto Wdnew   = std::make_shared<Function>(W);
	auto Wdcur   = std::make_shared<Function>(W);
	auto Wpcur   = std::make_shared<Function>(W);

	if(nstep == 0)
	{
		// Read initial condition from user define function:
		InitialConditions win;
		*Wpcur = win;
	}
	else
	{
		// Read initial condition from previous time step:
		std::string wpua ="sdata/Wdcur"+std::to_string(nstep)+".h5";
		std::string wpub ="sdata/Wpcur"+std::to_string(nstep)+".h5";

		auto f1 = HDF5File(MPI_COMM_WORLD,wpua, "r");
		f1.read(*Wdcur,  "Wdcur");
		f1.close();

		auto f2 = HDF5File(MPI_COMM_WORLD,wpub, "r");
		f2.read(*Wpcur,  "Wpcur");
		f2.close();
	}

	// Extract the three components for current time step
	auto du0  = std::make_shared<Function>((*Wdcur)[0]);
	auto p0   = std::make_shared<Function>((*Wdcur)[1]);
	auto dphi0= std::make_shared<Function>((*Wdcur)[2]);

	// Extract the three components for next time step
	auto du1  = std::make_shared<Function>((*Wdnew)[0]);
	auto p1   = std::make_shared<Function>((*Wdnew)[1]);
	auto dphi1= std::make_shared<Function>((*Wdnew)[2]);

	// Extract the two components for current time step
	auto u0   = std::make_shared<Function>((*Wpcur)[0]);
	auto phi0 = std::make_shared<Function>((*Wpcur)[2]);

	// Function for geometric redistancing
	auto phid0  = std::make_shared<Function>(Wrd);
	auto phid1  = std::make_shared<Function>(Wrd);

	double tinc= tinc_;
	double tinv= 1.0/tinc;
	double pse_tinc = pse_dt_;
	double pse_tinv = 1.0/pse_tinc;
	int    ttol= 100000;

	// Constant variable used for calculate fraction of liquid:
	auto zerop= std::make_shared<Constant>(0.0);
	auto onep = std::make_shared<Constant>(1.0);
	auto twop = std::make_shared<Constant>(2.0);

	// Initial variable needed for variational formulation:
	auto rhoa = std::make_shared<Constant>(rhoa_);
	auto rhow = std::make_shared<Constant>(rhow_);
	auto mua  = std::make_shared<Constant>(mua_);
	auto muw  = std::make_shared<Constant>(muw_);

	auto k      = std::make_shared<Constant>(tinc);
	auto idt    = std::make_shared<Constant>(tinv);
	auto pse_k  = std::make_shared<Constant>(pse_tinc);
	auto pse_idt= std::make_shared<Constant>(pse_tinv);
	auto fx     = std::make_shared<Source>();

	auto epslen  = std::make_shared<Constant>(epslen_);
	auto epsset  = std::make_shared<Constant>(epsset_);
	auto lc_kdc  = std::make_shared<Constant>(lc_kdc_);
	auto phid_kdc= std::make_shared<Constant>(phid_kdc_);
	auto phid_penalty = std::make_shared<Constant>(phid_penalty_);
	
	auto phi_top = std::make_shared<Phitop>(0.0);

	// Collect coefficient ns and levelset convection
	// Assign the parameters in Residual and Jacobian
	std::map<std::string, std::shared_ptr<const GenericFunction>> coefficients_F
	= {{"Wn",Wdnew},{"u0",u0},{"du0",du0},{"phi0",phi0},{"dphi0",dphi0},\
	{"fx",fx},{"ur",ur},{"k",k},{"idt",idt},{"epslen",epslen},{"lc_kdc",lc_kdc},\
	{"phi_top",phi_top},\
	{"rhoa",rhoa},{"rhow",rhow},{"mua",mua},{"muw",muw}};

	std::map<std::string, std::shared_ptr<const GenericFunction>> coefficients_J
	= {{"Wn",Wdnew},{"u0",u0},{"du0",du0},{"phi0",phi0},{"dphi0",dphi0},\
	{"fx",fx},{"ur",ur},{"k",k},{"idt",idt},{"epslen",epslen},{"lc_kdc",lc_kdc},\
	{"phi_top",phi_top},\
	{"rhoa",rhoa},{"rhow",rhow},{"mua",mua},{"muw",muw}};

	// Passing the mapping to header file:
	J->set_coefficients(coefficients_J);
	F->set_coefficients(coefficients_F);

	//============Mapping for preconditioning============//
	// Define dof mapping for velocity and pressure
	IS is[3];
	auto u_dofs = W->sub(0)->dofmap()->dofs();
	auto p_dofs = W->sub(1)->dofmap()->dofs();
	auto T_dofs = W->sub(2)->dofmap()->dofs();
	dolfin::cout << "Number of u and p and T dofs: " << u_dofs.size() << ", "
		<< p_dofs.size() << ", " << T_dofs.size() << dolfin::endl;
	ISCreateGeneral(PETSC_COMM_WORLD, u_dofs.size(), u_dofs.data(),
		PETSC_COPY_VALUES, &is[0]);
	ISCreateGeneral(PETSC_COMM_WORLD, p_dofs.size(), p_dofs.data(),
		PETSC_COPY_VALUES, &is[1]);
	ISCreateGeneral(PETSC_COMM_WORLD, T_dofs.size(), T_dofs.data(),
		PETSC_COPY_VALUES, &is[2]);
	std::vector<std::vector<dolfin::la_index>> dof_vec;
	dof_vec.push_back(u_dofs);
	dof_vec.push_back(p_dofs);
	dof_vec.push_back(T_dofs);

	PetscInt dofnum;
	dofnum = u_dofs.size() + p_dofs.size() + T_dofs.size();
	PetscSection section;
	PetscSectionCreate(PETSC_COMM_WORLD, &section);
	PetscSectionSetNumFields(section, 3);
	PetscSectionSetChart(section, 0, dofnum);

	std::vector<std::string> name_list;
	name_list.push_back("0");
	name_list.push_back("1");
	name_list.push_back("2");

	PetscInt goffset;
	goffset = u_dofs[0];
	for(int idof=0;idof<3;idof++){
		for(int jj=0;jj<dof_vec[idof].size();jj++){
			if(dof_vec[idof][jj]<goffset){
				goffset=dof_vec[idof][jj];
			}
		}
	}
	
	for(int ii=0;ii<3;ii++){
		std::string fkname = name_list[ii];
		PetscSectionSetFieldName(section,ii,fkname.c_str());
		for(int jj=0;jj<dof_vec[ii].size();jj++){
			PetscSectionSetDof(section, dof_vec[ii][jj]-goffset, 1);
			PetscSectionSetFieldDof(section, dof_vec[ii][jj]-goffset, ii, 1);
		}
	}
	PetscSectionSetUp(section);
	DM shell;
	DMShellCreate(PETSC_COMM_WORLD, &shell);
	DMSetDefaultSection(shell,section);
	DMSetUp(shell);
	//============Mapping for preconditioning============//

	//============fluid and levelset convection solver============//
	// Create Krylov Solver with GMRES method here:
	auto solver = std::make_shared<PETScKrylovSolver>("gmres");
	KSP ksp = solver->ksp();
	solver->parameters["error_on_nonconvergence"] = false;
	solver->parameters["relative_tolerance"]      = 1.0e-7;
	PC pc;

	KSPGetPC(ksp, &pc);
	KSPSetDM(ksp, shell);
	KSPSetDMActive(ksp, PETSC_FALSE);

	dolfin::PETScOptions::set("ksp_view");
	dolfin::PETScOptions::set("ksp_monitor_true_residual");
	dolfin::PETScOptions::set("ksp_pc_side", "right");
	dolfin::PETScOptions::set("ksp_max_it","80");

	/*
	dolfin::PETScOptions::set("pc_type", "hypre");
	dolfin::PETScOptions::set("pc_hypre_type", "boomeramg");
	dolfin::PETScOptions::set("pc_hypre_boomeramg_coarsen_type", "pmis");
	dolfin::PETScOptions::set("pc_hypre_boomeramg_interp_type", "FF1");
	*/

	dolfin::PETScOptions::set("pc_type", "fieldsplit");
	dolfin::PETScOptions::set("pc_fieldsplit_type", "additive");
	dolfin::PETScOptions::set("pc_fieldsplit_0_fields", "0,1");
	dolfin::PETScOptions::set("pc_fieldsplit_1_fields", "2");
	dolfin::PETScOptions::set("fieldsplit_0_pc_type", "fieldsplit");
	dolfin::PETScOptions::set("fieldsplit_0_pc_fieldsplit_type", "schur");
	dolfin::PETScOptions::set("fieldsplit_0_pc_fieldsplit_schur_fact_type", "upper");
	dolfin::PETScOptions::set("fieldsplit_0_fieldsplit_0_ksp_type", "preonly");
	dolfin::PETScOptions::set("fieldsplit_0_fieldsplit_0_pc_type", "jacobi");
	dolfin::PETScOptions::set("fieldsplit_0_fieldsplit_1_ksp_type", "preonly");
	dolfin::PETScOptions::set("fieldsplit_0_fieldsplit_1_pc_type", "hypre");
	dolfin::PETScOptions::set("fieldsplit_0_fieldsplit_1_pc_hypre_type", "boomeramg");
	dolfin::PETScOptions::set("fieldsplit_0_fieldsplit_1_pc_hypre_boomeramg_coarsen_type", "pmis");
	dolfin::PETScOptions::set("fieldsplit_0_fieldsplit_1_pc_hypre_boomeramg_interp_type", "FF1");
	dolfin::PETScOptions::set("fieldsplit_0_fieldsplit_1_pc_hypre_boomeramg_strong_threshold", "0.5");	
	dolfin::PETScOptions::set("fieldsplit_2_ksp_type", "preonly");
	dolfin::PETScOptions::set("fieldsplit_2_pc_type",  "jacobi");
	KSPSetFromOptions(ksp);
	//============fluid and levelset convection solver============//

	/*
	//============rd solver============//
	auto solver_rd = std::make_shared<PETScKrylovSolver>("gmres");
	KSP ksp_rd = solver_rd->ksp();
	solver_rd->parameters["error_on_nonconvergence"] = false;
	PC pc_rd;

	// Use fieldsplit for preconditioning
	KSPGetPC(ksp_rd, &pc_rd);

	dolfin::PETScOptions::set("ksp_view");
	dolfin::PETScOptions::set("ksp_monitor_true_residual");
	dolfin::PETScOptions::set("ksp_pc_side", "right");
	// AMG is chosen for preconditioning
	dolfin::PETScOptions::set("pc_type", "hypre");
	dolfin::PETScOptions::set("pc_hypre_type", "boomeramg");
	// Long range interpolation is crucial for efficiency of AMG
	dolfin::PETScOptions::set("pc_hypre_boomeramg_coarsen_type", "pmis");
	dolfin::PETScOptions::set("pc_hypre_boomeramg_interp_type", "FF1");
	dolfin::PETScOptions::set("pc_hypre_boomeramg_strong_threshold", "0.5");
	KSPSetFromOptions(ksp_rd);
	//============rd solver============//
	*/

	auto A_rd = std::make_shared<PETScMatrix>();
	auto b_rd = std::make_shared<PETScVector>();

	// Nonlinear problem define
	EquationSol nseq(F,   J,   bcs);
	//EquationSol rdeq(Frd, Jrd, bcs_rd);

	// Newton solver define
	NewtonSolver newton_solver(MPI_COMM_WORLD, solver, PETScFactory::instance());
	newton_solver.parameters["relative_tolerance"] = 1.0e-4;
	newton_solver.parameters["absolute_tolerance"] = 1.0e-8;
	// Solver do not crash even when the Newton Solver does not converge
	newton_solver.parameters["error_on_nonconvergence"] =false;
	// Maximum Newton iteration
	newton_solver.parameters["maximum_iterations"] = 4;

	// Define string used for saving data:
	std::string name_save,name_start,name_end;
	name_start = "res/T";
	name_end   = ".pvd";

	// Parameters for general alpha method
	double rhoc = 0.5;
	double am   = 0.5*(3.0-rhoc)/(1.0+rhoc);
	double af   = 1.0/(1+rhoc);
	double gamma= 0.5 + am -af;
	double c1   = tinc*(1-gamma);
	double c2   = tinc*gamma;

	std::string velname   ="results/vel";
	std::string prename   ="results/pre";
	std::string phiname   ="results/phi";
	std::string e1name    ="d.pvd";
	std::string sname, mname, nname;

	std::string waname ="sdata/Wdcur";
	std::string wbname ="sdata/Wpcur";
	std::string e2name =".h5";
	std::string was,wbs;

	// Redistance prep
	redistance redist;
	redist.redistance_prep(mesh);

	while (tnum < ttol)
	{

		//=============Solving NS and level set convection =============//
		// Initial Guess and Apply acceleration boundary conditions
		du1  = std::make_shared<Function>((*Wdcur)[0]);
		p1   = std::make_shared<Function>((*Wdcur)[1]);
		dphi1= std::make_shared<Function>((*Wdcur)[2]);
		*(du1->vector())   *= (gamma-1.0)/gamma;
		*(dphi1->vector()) *= (gamma-1.0)/gamma;
		assign(Wdnew,{du1,p1,dphi1});
		for (std::size_t i = 0; i < bcs_der.size(); i++)
			bcs_der[i]->apply(*Wdnew->vector());

		// Solve the Fixed point iterations
		newton_solver.solve(nseq, *Wdnew->vector());

		// Calculate the new velocity and Apply velocity boundary conditions
		Wpcur->vector()->axpy(c1, *Wdcur->vector());
		Wpcur->vector()->axpy(c2, *Wdnew->vector());
		for (std::size_t i = 0; i < bcs_ori.size(); i++)
			bcs_ori[i]->apply(*Wpcur->vector());

		// Save velocity:
		*Wdcur->vector() = *Wdnew->vector();

		// Update the derivative of velocity and phi
		du1  = std::make_shared<Function>((*Wdnew)[0]);
		assign(du0,du1);
		dphi1= std::make_shared<Function>((*Wdnew)[2]);
		assign(dphi0,dphi1);

		// Update the velocity and phi
		du1  = std::make_shared<Function>((*Wpcur)[0]);
		assign(u0,du1);
		dphi1= std::make_shared<Function>((*Wpcur)[2]);
		assign(phi0,dphi1);
		//=============Solving NS and level set convection =============//

		
		//=============Start geometric redistancing===========//
		{
			assign(phid0,   phi0);
			assign(phid1,   phi0);

			// Redistancing by geometric calculation
			redist.step_num = tnum;
			redist.geo_redistance(mesh,phid0,d2v_map,v2d_map);

			// Update variables
			auto utmp  = std::make_shared<Function>((*Wpcur)[0]);
			auto ptmp  = std::make_shared<Function>((*Wpcur)[1]);
			assign(Wpcur, {utmp, ptmp, phid0});
			assign(phi0, phid0);
		}
		//=============End geometric redistancing=============//
		

		// Save velocity
		if (tnum%10000==0)
		{
			sname = velname+std::to_string(tnum)+e1name;
			File ufile(sname);
			auto u1 = (*Wpcur)[0];
			ufile << u1;

			nname = prename+std::to_string(tnum)+e1name;
			File Pfile(nname);
			auto Ps = (*Wdcur)[1];
			Pfile << Ps;

			mname = phiname+std::to_string(tnum)+e1name;
			File phifile(mname);
			auto phis = (*Wpcur)[2];
			phifile << phis;
		}

		if (tnum%10==0)
		{
			was     = waname+std::to_string(tnum)+e2name;
			wbs     = wbname+std::to_string(tnum)+e2name;
			auto fout1 = HDF5File(MPI_COMM_WORLD,was, "w");
			fout1.write(*Wdcur,"Wdcur");
			auto fout2 = HDF5File(MPI_COMM_WORLD,wbs, "w");
			fout2.write(*Wpcur,"Wpcur");
			
			if(rankvalues == 0){
				std::string cof_name = "step.dat";
				std::ofstream stFile(cof_name);
				stFile << tnum ;
				stFile << "\n";
				stFile.close();
			}
		}

		if(rankvalues == 0) std::cout<<"This is step "<<tnum<<std::endl;

		// Time variable increment here:
		time += tinc;
		tnum  = tnum + 1;

		phi_top->_t = time;

	}

	return 0;
}
