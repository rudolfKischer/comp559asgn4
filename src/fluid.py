## Rudolf Kischer 260956107

import moderngl_window as mglw
import moderngl as mgl
import numpy as np
import glm as glm
# break point for debugger
import pdb 
from numba import jit

# @jit(nopython=True)
def gauss_seidel_poisson(p, div, iterations, tolerance=1e-5):
    p_new = np.zeros_like(p)
    for i in range(iterations):
        p_new[1:-1, 1:-1] = (div[1:-1, 1:-1] +
                      p[:-2, 1:-1] + p[2:, 1:-1] + 
                      p[1:-1, :-2] + p[1:-1, 2:]) / 4
        # get total diff
        diff = np.sum(np.abs(p_new - p)) 
        p[:] = p_new
        # print('diff:', diff, 'its:', i)
        if diff < tolerance:
            break
    return p

# @jit(nopython=True)
def gauss_seidel_diffuse(q0, q1, a, its, tolerance=1e-5):
    '''Gauss-Seidel solver for diffusion equation with early stopping.'''
    q_temp = np.zeros_like(q1)
    for i in range(its):
        q_temp[1:-1, 1:-1] = (q0[1:-1, 1:-1] + a * (q1[:-2, 1:-1] + q1[2:, 1:-1] + q1[1:-1, :-2] + q1[1:-1, 2:])) / (1 + 4 * a)
        diff = np.sum(np.abs(q1 - q_temp)) 
        q1[:] = q_temp
        # print('diff:', diff, 'its:', i)
        if diff < tolerance:
            break
    return q1







class Source:
  '''A source of temperature (heating or cooling) in the fluid.'''
  def __init__(self, x, y, strength):
    self.x = x
    self.y = y
    self.strength = strength
    if strength > 0: self.colour = (1, 0, 0, 1)
    else: self.colour = (0, 0, 1, 1)

class HelloWorld(mglw.WindowConfig):
  '''
  Stable Fluid Simulation

  Note that array indexing is row column, so the first index is the y index and the second index is the x index.
  Note that there is an extra layer of cells outside the domain, so integer indexing within the actual domain is from 1 to nx, 1 to ny.
  Note that the velocities are stored at the centers of the cells.
  The spatial limits of the domain are given by xl xh yl yh.
  '''

  # nx = 32 # TODO: Change the resolution to different sizes in testing!
  # ny = 16
  nx = 32
  ny = 16


  pix_per_cell = 1024/(nx+4) # Set the cell size to get a nice window size 
  win_width = pix_per_cell*(nx+4)
  win_height = pix_per_cell*(ny+4)
  gl_version = (3, 3)
  title = "Stable Fluid Simulation - Rudolf C. Kischer"
  window_size = (win_width, win_height)
  aspect_ratio = win_width / win_height
  resizable = True
  resource_dir = 'data'

  color_samples = 5 # sample temperature colours at a higher resolution than the grid for better visualisation

  wrapping_boundary = True

  yl = -1.0
  yh =  1.0
  dy = 2.0/ny
  dx = dy
  xl = -dx*nx/2.0 
  xh =  dx*nx/2.0 

  mouse_strength = 2.1
  source_strength = 0.2

  gravity_enabled = False
  gravity = 0.1

  prev_mouse_pos = None

  draw_grid_enabled = True
  draw_grid_lines_enabled = True
  draw_velocity_enabled = True
  velocity_scale = 0.1
  step_request = False
  reset_request = False
  running = False
  num_particles = 10000

  point_size = 3.0

  iterations = 10

  dt = 0.01    # time step (control with up and down arrows)
  viscosity = 0.01 # viscosity of fluid (diffuision of velocities) (control with 1 and 2 keys)
  kappa = 0.001#0.000001      # thermal diffusivity (control with 3 and 4 keys)
  beta = 0.05         # buoyancy force control (control with 5 and 6 keys)

  def __init__(self, **kwargs):
    super().__init__(**kwargs)  
    self.ctx.enable(mgl.BLEND)
    self.ctx.blend_func = mgl.SRC_ALPHA, mgl.ONE_MINUS_SRC_ALPHA
    self.ctx.point_size = self.point_size
    self.prog = self.ctx.program( vertex_shader = open('glsl/vert.glsl').read(), fragment_shader = open('glsl/frag.glsl').read() )
    self.prog2 = self.ctx.program( vertex_shader = open('glsl/constantColourVert.glsl').read(), fragment_shader = open('glsl/constantColourFrag.glsl').read() )
    # transformation for drawing the fluid grid
    self.MVP = glm.ortho( self.xl-self.dx*2, self.xh+self.dx*2, self.yl-self.dy*2, self.yh+self.dy*2, -1, 1)
    self.prog['MVP'].write( self.MVP )
    self.prog2['MVP'].write( self.MVP )
    self.star_vao = self.load_scene("star.obj").root_nodes[0].mesh.vao.instance(self.prog2) 
    self.reset()
    self.setup_grid_lines()
    self.setup_velocity_lines()
    self.setup_boundary_box()
    self.setup_color_mesh()
    self.setup_particles()
    self.sources = []

  def reset(self):


    self.curr_v = np.zeros((self.ny+2, self.nx+2, 2)) # velocities
    self.next_v = np.zeros((self.ny+2, self.nx+2, 2)) # velocities
    self.curr_tp = np.zeros((self.ny+2, self.nx+2)) # temperature
    self.next_tp = np.zeros((self.ny+2, self.nx+2)) # temperature
    # reset the particles
    self.reset_particles()

    # remove sources ans sinks
    self.sources = []


  def xy_to_ij(self, x, y):
    '''Converts x,y position to i,j indices and bilinear interpolation coefficients a,b.'''
    # Accounting for padding one expects at xl, yl, i=0, j=0 with a=b=0.5
    j = np.floor((x-self.xl)/self.dx+0.5).astype(int)
    i = np.floor((y-self.yl)/self.dy+0.5).astype(int)
    a = (x-self.xl)/self.dx+0.5 - j
    b = (y-self.yl)/self.dy+0.5 - i
    return i, j, a, b
  
  def ij_to_xy(self, i, j):
    '''Converts i,j indices to x,y position.'''
    return self.xl + j*self.dx, self.yl + i*self.dy

  def np_xy_to_ij(self, XY):
    '''converts an array of length n of xy positions to ij coordinates and returns a n x 4 array of i, j, a, b'''
    x = XY[:,0]
    y = XY[:,1]
    j = np.floor((x-self.xl)/self.dx+0.5).astype(int)
    i = np.floor((y-self.yl)/self.dy+0.5).astype(int)
    a = (x-self.xl)/self.dx+0.5 - j
    b = (y-self.yl)/self.dy+0.5 - i
    return np.column_stack((i,j,a,b))


  def setup_color_mesh(self):
    ''' Sets up drawing of a coloured mesh on the domain to show temperatures '''
    # Note the mesh resolution is higher than the grid resolution for better visualisation
    x,y = np.meshgrid(np.linspace(self.xl,self.xh,self.nx*self.color_samples), np.linspace(self.yl,self.yh,self.ny*self.color_samples))
    self.cg_i,self.cg_j,self.cg_a,self.cg_b = self.xy_to_ij(x,y) # store color grid (cg) indices and interpolation coefficients
    vertices = np.column_stack((x.flatten(), y.flatten())).astype('f4')
    # Triangle strip would be better, but just making *ALL* the triangles is easier
    ind = np.reshape( range(x.size), (self.ny*self.color_samples,self.nx*self.color_samples))
    j,i = np.meshgrid( range(self.nx*self.color_samples-1), range(self.ny*self.color_samples-1))
    indices1 = np.column_stack((ind[i, j].flatten(), ind[i, j + 1].flatten(), ind[i + 1, j].flatten()))
    indices2 = np.column_stack((ind[i, j + 1].flatten(), ind[i + 1, j + 1].flatten(), ind[i + 1, j].flatten()))
    indices = np.concatenate( (indices1, indices2), axis=0 )
    vbo_pos = self.ctx.buffer(vertices.astype('f4').tobytes())
    self.vbo_col = self.ctx.buffer(np.zeros((x.size,4)).astype('f4').tobytes())  
    ibo = self.ctx.buffer(indices.astype("i4").tobytes())  
    self.grid_vao = self.ctx.vertex_array( self.prog,
      [ (vbo_pos, '2f', 'in_position'), (self.vbo_col, '4f', 'in_colour') ], index_buffer=ibo, mode=mgl.TRIANGLES )

  def draw_color_mesh(self):
    ''' Updates the temperature mesh colours and draws it. '''
    i,j,a,b = self.cg_i,self.cg_j,self.cg_a,self.cg_b 
    ## bilinear interpolation
    value = (1-a)*(1-b)*self.curr_tp[i,j] + (1-a)*b*self.curr_tp[i+1,j] + a*(1-b)*self.curr_tp[i,j+1] + a*b*self.curr_tp[i+1,j+1]
    colours = np.zeros((i.size,4), dtype='f4')
    colours[:,0] = np.clip(value.flatten(), 0, 1)
    colours[:,2] = np.clip(-value.flatten(), 0, 1)
    colours[:,3] = 1
    self.vbo_col.write(colours.astype("f4").tobytes())
    self.grid_vao.render()

  def setup_velocity_lines(self):
    '''Create a vertex array with repeated mesh grid values for drawing velocity lines.'''
    x = np.linspace( self.xl-0.5*self.dx, self.xh+0.5*self.dx, self.nx+2)
    y = np.linspace( self.yl-0.5*self.dy, self.yh+0.5*self.dy, self.ny+2)
    xg, yg = np.meshgrid(x,y)    
    self.vertices = np.zeros((xg.size*2,2), dtype='f4')  # *2 for center and center plus velocity
    self.vertices[0::2,0] = xg.flatten()
    self.vertices[0::2,1] = yg.flatten()
    self.vertices[1::2,0] = xg.flatten() 
    self.vertices[1::2,1] = yg.flatten() 
    self.vbo = self.ctx.buffer( self.vertices.astype("f4").tobytes(),dynamic=True )
    self.lines_vao = self.ctx.vertex_array(self.prog2, [(self.vbo, '2f', 'in_position')], mode=mgl.LINES)
  
  def draw_velocity_lines(self):
    '''Draw velocities after updating the line segment enpoints in the dynamic draw vbo of positions'''

    horizontal_v = self.velocity_scale*self.curr_v[:,:,0].flatten()
    vertical_v = self.velocity_scale*self.curr_v[:,:,1].flatten()

    max_mag = 10*self.dx

    # horizontal_v = np.clip(horizontal_v, -max_mag, max_mag)
    # vertical_v = np.clip(vertical_v, -max_mag, max_mag)

    self.vertices[1::2,0] = self.vertices[0::2,0] + horizontal_v
    self.vertices[1::2,1] = self.vertices[0::2,1] + vertical_v

    self.vbo.write(self.vertices.astype("f4").tobytes())
    self.lines_vao.program['colour'] = (0, 1, 1, 0.5)
    self.lines_vao.render()

  def reset_particles(self):
    '''Randomly distribute particles in the domain.'''
    x = np.random.uniform(self.xl, self.xh, self.num_particles)
    y = np.random.uniform(self.yl, self.yh, self.num_particles)
    self.particles = np.column_stack((x, y))

  def setup_particles(self):
    self.reset_particles()
    self.particle_vbo = self.ctx.buffer(self.particles.astype('f4').tobytes())
    self.particle_vao = self.ctx.vertex_array(self.prog2, [(self.particle_vbo, '2f', 'in_position')], mode=mgl.POINTS)
  
  def draw_particles(self):
    self.particle_vbo.write(self.particles.astype('f4').tobytes())
    # get the colour by sampling the temperature at the particle positions

    self.particle_vao.program['colour'] = (1, 1, 1, 1)
    self.particle_vao.render()

  def setup_grid_lines(self):
    '''Creates a vao for drawing the grid lines (note there is one layer of cells outside the domain)'''
    # have nx*ny cells, but padding by one extra on all sides if not using MAC grid
    verticesy = np.zeros(((self.ny+3)*2,2), dtype='f4') # set vertices to be horizontal lines
    verticesy[0::2,0] = self.xl - self.dx
    verticesy[0::2,1] = np.linspace(self.yl-self.dy, self.yh+self.dy, self.ny+3)
    verticesy[1::2,0] = self.xh + self.dx
    verticesy[1::2,1] = np.linspace(self.yl-self.dy, self.yh+self.dy, self.ny+3)
    verticesx = np.zeros(((self.nx+3)*2,2), dtype='f4') # set vertices to be vertical lines
    verticesx[0::2,0] = np.linspace(self.xl-self.dx, self.xh+self.dx, self.nx+3)
    verticesx[0::2,1] = self.yl - self.dy
    verticesx[1::2,0] = np.linspace(self.xl-self.dx, self.xh+self.dx, self.nx+3)
    verticesx[1::2,1] = self.yh + self.dy
    vertices = np.concatenate((verticesx, verticesy), axis=0) # concatenate the two arrays
    vbo_pos = self.ctx.buffer(vertices.astype('f4').tobytes())
    self.grid_lines_vao = self.ctx.vertex_array( self.prog2, [ (vbo_pos, '2f', 'in_position')], mode=mgl.LINES)

  def setup_boundary_box(self):
    '''Creates a vao for drawing a box showing the domain (note there is one layer of cells outside the domain)'''
    vertices = np.array([[self.xl, self.yl],[self.xl, self.yh],[self.xh, self.yh],[self.xh, self.yl]])
    vbo_pos = self.ctx.buffer(vertices.astype('f4').tobytes())
    self.domain_box_vao = self.ctx.vertex_array( self.prog2, [ (vbo_pos, '2f', 'in_position')], mode=mgl.LINE_LOOP)

  def key_event(self, key, action, modifiers):
    if action == self.wnd.keys.ACTION_PRESS:
      if key == self.wnd.keys.V: self.draw_velocity_enabled = not self.draw_velocity_enabled  
      if key == self.wnd.keys.G: self.draw_grid_lines_enabled = not self.draw_grid_lines_enabled  
      if key == self.wnd.keys.C: self.draw_grid_enabled = not self.draw_grid_enabled  
      if key == self.wnd.keys.P: self.reset_particles()
      if key == self.wnd.keys.R: self.reset_request = True
      if key == self.wnd.keys.S: self.step_request = True  
      if key == self.wnd.keys.SPACE: self.running = not self.running  
      if key == self.wnd.keys.ESCAPE: self.close()
      if key == self.wnd.keys.LEFT: self.velocity_scale *= 0.5
      if key == self.wnd.keys.RIGHT: self.velocity_scale *= 2.0
      if key == self.wnd.keys.COMMA: 
        self.iterations = np.clip(self.iterations/2, 2, 256).astype(int)
        print("Iterations:", self.iterations)
      if key == self.wnd.keys.PERIOD: 
        self.iterations = np.clip(self.iterations*2, 2, 256).astype(int)
        print("Iterations:", self.iterations)
      if key == self.wnd.keys.UP: 
        self.dt *= 2.0
        print("dt:", self.dt)
      if key == self.wnd.keys.DOWN: 
        self.dt *= 0.5
        print("dt:", self.dt)
      if key == self.wnd.keys.NUMBER_1: 
        self.viscosity *= 0.5
        print("Viscosity:", self.viscosity)
      if key == self.wnd.keys.NUMBER_2: 
        self.viscosity *= 2.0
        print("Viscosity:", self.viscosity)
      if key == self.wnd.keys.NUMBER_3:
        self.kappa *= 0.5
        print("Thermal diffusivity:", self.kappa)
      if key == self.wnd.keys.NUMBER_4:
        self.kappa *= 2.0
        print("Thermal diffusivity:", self.kappa)
      if key == self.wnd.keys.NUMBER_5:
        self.beta *= 0.5
        print("Buoyancy force control:", self.beta)
      if key == self.wnd.keys.NUMBER_6:
        self.beta *= 2.0
        print("Buoyancy force control:", self.beta)
      if key == self.wnd.keys.DELETE:
        self.sources = []
        
  def mouse_press_event(self, x, y, button):
    xx, yy = self.mouse_to_xy(x, y)
    if (xx < self.xl or xx > self.xh or yy < self.yl or yy > self.yh): return
    if button == self.wnd.mouse.left:
      self.sources.append(Source(xx, yy, -self.source_strength))
    if button == self.wnd.mouse.right:
      self.sources.append(Source(xx, yy, self.source_strength))

  def mouse_to_xy(self, x, y):
    '''Converts mouse coordinates to x,y position in the domain.'''
    (w,h) = self.wnd.size
    self.pix_per_cell = np.min((w/(self.nx+4), h/(self.ny+4)))
    xx = (x-w/2)/self.pix_per_cell*self.dx
    yy = (h/2-y)/self.pix_per_cell*self.dy
    return xx, yy

  def mouse_position_event(self, mouse_x, mouse_y, mouse_dx, mouse_dy):
    ''' Applies a force to the velocity grid. '''
    if self.prev_mouse_pos is None:
      self.prev_mouse_pos = (mouse_x, mouse_y)
      return

    xx, yy = self.mouse_to_xy(mouse_x, mouse_y)
    if (xx < self.xl or xx > self.xh or yy < self.yl or yy > self.yh): return
    i,j,a,b = self.xy_to_ij(xx,yy)

    dv = self.mouse_strength  * np.array([mouse_dx, -mouse_dy])

    # effect_radius = 0.1






    # loop over effect radious / dx and dy
    # if that point is in the domain, apply the force
    # max_rad_idx = self.xy_to_ij(xx + effect_radius, yy + effect_radius)
    # min_rad_idx = self.xy_to_ij(xx - effect_radius, yy - effect_radius)
    # # clamp
    # max_rad_idx = (np.clip(max_rad_idx[0], 0, self.ny), np.clip(max_rad_idx[1], 0, self.nx))
    # min_rad_idx = (np.clip(min_rad_idx[0], 0, self.ny), np.clip(min_rad_idx[1], 0, self.nx))

    # # divide dv by the number of cells that receive the force
    # # this is the area of the circle of effect time the square that is looped over
    # # dv = dv / ( (max_rad_idx[0] - min_rad_idx[0]) * (max_rad_idx[1] - min_rad_idx[1]) )

    # for p in range(min_rad_idx[0], max_rad_idx[0]):
    #   for k in range(min_rad_idx[1], max_rad_idx[1]):
    #     # if not in the manhatan distance of the point, continue
    #     # convert idx to xy
    #     diff_vec = np.array([ (p-i) * self.dy, (k-j) * self.dx ])
    #     dist = np.linalg.norm(diff_vec)
    #     if dist > effect_radius:
    #       continue

    #     # we want it to be strongest if its in the direction of the dv
    #     # we want it to be weakes if its in the opposite direction of the dv
    #     # to do this we will use the dot product of the direction of the dv and the direction of the point
    #     dv_D = dv / np.linalg.norm(dv) if np.linalg.norm(dv) != 0 else dv
    #     self.curr_v[p,k,:] += dv * (dist / effect_radius)


    if ( i >= 0 and i <= self.ny and j >= 0 and j <= self.nx):
      self.curr_v[i,j,:] += (1-a)*(1-b)*dv
      self.curr_v[i+1,j,:] += (1-a)*b*dv
      self.curr_v[i,j+1,:] += a*(1-b)*dv
      self.curr_v[i+1,j+1,:] += a*b*dv

    self.set_boundary(self.curr_v[:,:,0], "horizontal")
    self.set_boundary(self.curr_v[:,:,1], "vertical")

    # set prev mouse pos to current mouse pos

    # print the values of vertical and horizontal the boundaries

  # Code below this point is the fluid simulation code  

  def diffuse(self, q0, kappa, dt, boundary_type):
    # use gauss-seidel to solve the linear system
    # stable diffusion
    # backward euler stop
    # kappa : diffusion constant
    a = dt * kappa * (self.nx * self.ny)
    q_1 = np.copy(q0)

    q_1 = gauss_seidel_diffuse(q0, q_1, a, self.iterations)
    self.set_boundary(q_1, boundary_type)
    
    return q_1

  def set_boundary(self, q, boundary_type):
    ''' 
    Set boundary conditions for the given quantity q.
    Parameters:
      q (np.array): Quantity to set boundary conditions for
      boundary_type (str): "vertical" forces the quantity to be antisymmetric at the top and bottom boundaries, 
                          "horizontal" forces the quantity to be antisymmetric at the left and right boundaries
    '''
    # if boundary type is none, set the quantity at the boundaries equal to their neighbours in the boundary
    # q is a 2D array of scalars
    # note the boundary is equal to one cell of padding on all sides
    # if None, we will copy the value of the neighbouring cell in the appropriate direction

    # make the boundaries wrap instead, so copy the value of the opposite boundary
    if self.wrapping_boundary:
      q[0,:] = q[-2,:]
      q[-1,:] = q[1,:]
      q[:,0] = q[:,-2]
      q[:,-1] = q[:,1]
      return
    if boundary_type is None:
      q[0,:] = q[1,:]
      q[-1,:] = q[-2,:]
      q[:,0] = q[:,1]
      q[:,-1] = q[:,-2]

    # # FOR SOME REASON IF ITS IN THESE IF STATEMENTS IT DOESNT WORK
    # # IT RUNS BUT THE BOUNDARY CONDITIONS ARENT SET
    
    if "vertical" is boundary_type:
      # horizontal component should be 0 on the vertical boundaries
      # apply antisymmetric boundary conditions
      # q[0,:] = -q[1,:]
      # q[-1,:] = -q[-2,:]
      q[0,:] = 0
      q[-1,:] = 0
    
    if "horizontal" is boundary_type:
      # vertical component should be 0 on the horizontal boundaries
      # apply antisymmetric boundary conditions
      q[:,0] = 0
      q[:,-1] = 0


    

    # return 
  
  def lerp(self, A, B, t):
    ''' Linear interpolation, but componentwise on two vectors '''
    return A*(1-t).reshape(-1, 1) + B*t.reshape(-1, 1)



  def advect_particles(self, dt):
    ''' Advects particles using the current grid velocities. '''
    # get the indices for the particles from their positions
    # get their  sub cell coordinates
    # get the velocities of the neighbouring cells
    # interpolate the velocities to the particle positions
    # us forward euler to update the particle positions
    # clip the particle positions to the domain
    # print(f'particles: {self.particles}')
    particles_ijab = self.np_xy_to_ij(self.particles)

    neighbourhood = np.array([[0,0],[0,1],[1,0],[1,1]])

    corner_indices = particles_ijab[:, None, :2] + neighbourhood[None, :, :]
    # ij returns the lower left corner of the cell
    # right now we have a n x 4 array of i,j,a,b
    # we want an n x 8 array of v0x, v0y, v1x, v1y, v2x, v2y, v3x, v3y
    # reshape corner indices to be n x 8 instead of n x 4 x 2

    # get the velocities of the neighbouring cells

    v_corners = self.curr_v[corner_indices[:,:,0].astype(int), corner_indices[:,:,1].astype(int)]

    # interpolate the velocities to the particle positions
    # v is n x 4 x 2
    # we want to interpolate between idx 0 and 1 and idx 2 and 3, and then interpolate between those two

    v_h_top = self.lerp(v_corners[:,0], v_corners[:,1], particles_ijab[:,2])
    v_h_bottom = self.lerp(v_corners[:,2], v_corners[:,3], particles_ijab[:,2])
    v = self.lerp(v_h_top, v_h_bottom, particles_ijab[:,3])

    # us forward euler to update the particle positions
    self.particles = self.particles + dt * v

    if self.gravity_enabled:
      self.particles[:,1] -= self.gravity * dt

    # wrap the particles around the domain
    if self.wrapping_boundary:
        self.particles[:,0] = np.mod(self.particles[:,0] - self.xl, self.xh - self.xl) + self.xl
        self.particles[:,1] = np.mod(self.particles[:,1] - self.yl, self.yh - self.yl) + self.yl


    # clip the particle positions to the domain
    # if a particle is outside the domain invert its velocity
        

    self.particles[:,0] = np.clip(self.particles[:,0], self.xl, self.xh)
    self.particles[:,1] = np.clip(self.particles[:,1], self.yl, self.yh)




    return 


  def advect(self, q, dt, gv, boundary_type):
    dt0 = dt * self.nx

    # Create a grid of coordinates for i and j
    j, i = np.meshgrid(np.arange(1, self.nx+1), np.arange(1, self.ny+1))

    # Calculate the departure points
    x = j - dt0 * gv[1:self.ny+1, 1:self.nx+1, 0]
    y = i - dt0 * gv[1:self.ny+1, 1:self.nx+1, 1]

    # Clip the departure points to lie within the domain
    # if wrapping is on we dont need to clip, we just need to wrap

    x = np.clip(x, 0.5, self.nx + 0.5)
    y = np.clip(y, 0.5, self.ny + 0.5)

        # x = np.clip(x, 0.5, self.nx + 0.5)
        # y = np.clip(y, 0.5, self.ny + 0.5)

    # Calculate the indices of the departure points
    i0 = np.floor(y).astype(int)
    j0 = np.floor(x).astype(int)
    i1 = i0 + 1
    j1 = j0 + 1


    # Calculate the bilinear interpolation weights
    s1 = y - i0
    s0 = 1 - s1
    t1 = x - j0
    t0 = 1 - t1

    # Perform bilinear interpolation
    q_1 = (s0 * (t0 * q[i0, j0] + t1 * q[i0, j1]) +
           s1 * (t0 * q[i1, j0] + t1 * q[i1, j1]))
    
    # pad
    q_1 = np.pad(q_1, ((1, 1), (1, 1)), mode='constant')


    # Apply boundary conditions
    self.set_boundary(q_1, boundary_type)

    return q_1
  
  # def advect(self, q, dt, gv, boundary_type):
  #   '''
  #   Advects given quantities q given the current grid velocities gv.
  #     Parameters:
  #       q (np.array): Quantities to be advected
  #       dt (float): Time step
  #       gv (np.array): Grid velocities
  #       boundary_type (str): type of boundary condition to apply (None, "vertical", "horizontal")
  #     Returns (np.array): Advected quantities
  #   '''

  #   dt0 = dt * self.nx
  #   q_1 = np.zeros_like(q)
  #   for i in range(1, self.ny+1):
  #     for j in range(1, self.nx+1):
  #       x = j - dt0 * gv[i,j,0]
  #       y = i - dt0 * gv[i,j,1]
  #       x = np.clip(x, 0.5, self.nx+0.5)
  #       y = np.clip(y, 0.5, self.ny+0.5)
  #       i0 = np.floor(y).astype(int)
  #       j0 = np.floor(x).astype(int)
  #       i1 = i0 + 1
  #       j1 = j0 + 1

  #       s1 = y - i0
  #       s0 = 1 - s1

  #       t1 = x - j0
  #       t0 = 1 - t1

  #       q_1[i,j] += s0*(t0*q[i0,j0] + t1*q[i0,j1]) + s1*(t0*q[i1,j0] + t1*q[i1,j1])
  #   self.set_boundary(q_1, boundary_type)


  #   return q_1

  def step(self, dt):
    self.add_source_temperature(dt)
    self.apply_temperature_force(dt)        
    self.velocity_step(dt)
    self.scalar_step(dt)
    self.advect_particles(dt)

  def add_source_temperature(self, dt):
    for source in self.sources:
      i,j,a,b = self.xy_to_ij(source.x, source.y)
      self.curr_tp[i,j] += (1-a)*(1-b)*source.strength
      self.curr_tp[i+1,j] += (1-a)*b*source.strength
      self.curr_tp[i,j+1] += a*(1-b)*source.strength
      self.curr_tp[i+1,j+1] += a*b*source.strength

  def apply_temperature_force(self, dt):
    '''Applies the buoyancy force to the velocity field.'''
    # compute the avg tempreture in all the cells
    avg_temp = np.mean(self.curr_tp)
    # compute the distance between the avg temp and the current temp at each cell
    diff_temp = self.curr_tp - avg_temp
    # apply the buoyancy force to the vertical velocity
    self.curr_v[:,:,1] += self.beta * diff_temp

    # apply boundary conditions
    self.set_boundary(self.curr_v[:,:,1], "horizontal")


    return # return nothing, this should modify the velocity field in place

  def scalar_step(self, dt):
    '''Advects and diffuses the temperature field.'''
    # modify the tempreture field in the horizontal direction
    self.next_tp = self.advect(self.curr_tp, dt, self.curr_v, None)
    self.curr_tp = self.diffuse(self.next_tp, self.kappa, dt, None)
    # modify the tempreture field in the vertical direction
    self.next_tp = self.advect(self.curr_tp, dt, self.curr_v, None)
    self.curr_tp = self.diffuse(self.next_tp, self.kappa, dt, None)


    return # return nothing, this should modify the temperature field in place
  
  from random import random
  

  def velocity_step(self, dt):
    # TODO: STEP 7: Complete this function with calls to the diffuse, project, and advect functions
    # diffuse_horizontal = self.diffuse(self.curr_v[:,:,0], self.viscosity, dt, "horizontal")
    # self.curr_v[:,:,0] = self.diffuse(diffuse_horizontal, self.viscosity, dt, "vertical")
    # diffuse_vertical = self.diffuse(self.curr_v[:,:,1], self.viscosity, dt, "vertical")
    # self.curr_v[:,:,1] = self.diffuse(diffuse_vertical, self.viscosity, dt, "horizontal")
    self.curr_v = self.diffuse(self.curr_v, self.viscosity, dt, None)
    self.project()

    # self.curr_v = self.advect(self.curr_v, dt, self.curr_v, None)
    # advect horizontal and then vertical
    self.curr_v[:,:,0] = self.advect(self.curr_v[:,:,0], dt, self.curr_v[:,:,:], None)
    self.project()
    self.curr_v[:,:,1] = self.advect(self.curr_v[:,:,1], dt, self.curr_v[:,:,:], None)
    # advect vertical and then horizontal
    self.project()

    # apply boundary conditions
    self.set_boundary(self.curr_v[:,:,0], "horizontal")
    self.set_boundary(self.curr_v[:,:,1], "vertical")

    return # return nothing, this should modify the velocity field in place

  # Projection step to make the velocity field divergence free
  def project(self):
    h = self.dx # grid spacing

    # Vectorize divergence calculation
    div = -0.5 * h * (self.curr_v[2:, 1:-1, 1] - self.curr_v[0:self.ny, 1:-1, 1] + 
                       self.curr_v[1:-1, 2:, 0] - self.curr_v[1:-1, 0:-2, 0])
    p = np.zeros_like(div)

    self.set_boundary(div, None)
    self.set_boundary(p, None)

    # Use Numba for optimizing Gauss-Seidel iterations
    p = gauss_seidel_poisson(p, div, self.iterations)  * 0.9

    # pad the pressure field
    p = np.pad(p, ((1, 1), (1, 1)), mode='constant')

    self.set_boundary(p, None)

    # Vectorize velocity correction
    self.curr_v[1:self.ny+1, 1:self.nx+1, 0] -= 0.5 * (p[1:-1, 2:] - p[1:-1, 0:-2]) / h
    self.curr_v[1:self.ny+1, 1:self.nx+1, 1] -= 0.5 * (p[2:, 1:-1] - p[0:-2, 1:-1]) / h

    self.set_boundary(self.curr_v[:,:,0], None)
    self.set_boundary(self.curr_v[:,:,1], None)


  # def project(self):
  #   '''Solves the pressure Poisson equation to make the velocity field divergence free.'''

  #   h = self.dx

  #   div = np.zeros((self.ny+2, self.nx+2))
  #   p =  np.zeros((self.ny+2, self.nx+2))

  #   for i in range(1, self.ny+1):
  #     for j in range(1, self.nx+1):
  #       div[i,j] = -0.5 * h * (self.curr_v[i+1,j,1] - self.curr_v[i-1,j,1] + self.curr_v[i,j+1,0] - self.curr_v[i,j-1,0])
  #       p[i,j] = 0
    
  #   self.set_boundary(div, None)
  #   self.set_boundary(p, None)


  #   for _ in range(self.iterations):
  #     for i in range(1, self.ny+1):
  #       for j in range(1, self.nx+1):
  #         p[i,j] = (div[i,j] + p[i-1,j] + p[i+1,j] + p[i,j-1] + p[i,j+1]) / 4
  #     self.set_boundary(p, None)
    
  #   for i in range(1, self.ny+1):
  #     for j in range(1, self.nx+1):
  #       self.curr_v[i,j,0] -= 0.5 * (p[i,j+1] - p[i,j-1]) / h
  #       self.curr_v[i,j,1] -= 0.5 * (p[i+1,j] - p[i-1,j]) / h
  #   self.set_boundary(self.curr_v[:,:,0], None)
  #   self.set_boundary(self.curr_v[:,:,1], None)


  #   return # return nothing, this should modify the velocity field in place
  
  def render(self, time, frame_time):    
    self.ctx.clear(0,0,0)
    if self.reset_request:
      self.reset_request = False
      self.reset()
    if self.running or self.step_request:
      self.step_request = False
      self.step(self.dt)    
    if self.draw_grid_enabled: 
      self.draw_color_mesh() 
    if self.draw_velocity_enabled: self.draw_velocity_lines()
    if self.draw_grid_lines_enabled: 
      self.grid_lines_vao.program['colour'] = (0.5, 0.5, 0.5, 0.25)
      self.grid_lines_vao.render()
    self.domain_box_vao.program['colour'] = (1, 1, 1, 1)
    self.domain_box_vao.render()
    self.draw_particles()
    for source in self.sources:
      self.prog2['colour'] = source.colour
      self.prog2['offset'] = (source.x, source.y)
      self.prog2['scale'] = (self.dx/4, self.dy/4)  
      self.star_vao.render()
    self.prog2['offset'] = (0,0) # reset to identity
    self.prog2['scale'] = (1,1)  # reset to identity      

HelloWorld.run()