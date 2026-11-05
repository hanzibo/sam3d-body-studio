# SPDX-License-Identifier: GPL-3.0-only
# Copyright (c) 2026 hanzibo and contributors.
"""Persistent GPU camera-image + calibrated body mesh display (no readback)."""
import ctypes
import traceback
import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets
from OpenGL import GL
from OpenGL.raw.GL.VERSION.GL_1_1 import glGenTextures as raw_gen_textures
from OpenGL.GL.shaders import compileProgram, compileShader
from mesh_geometry import vertex_normals
from pose_data import EDGES, COLORS, angle_text


def camera_projection(matrix, width, height, near=.01, far=1000.):
    k = np.asarray(matrix, dtype=np.float32)
    # Direct CV coordinates: x right, y down, z forward, and clip-w = z.
    return np.array([[2*k[0,0]/width, 2*k[0,1]/width, 2*k[0,2]/width-1, 0],
                     [0, -2*k[1,1]/height, 1-2*k[1,2]/height, 0],
                     [0, 0, (far+near)/(far-near), -2*far*near/(far-near)],
                     [0, 0, 1, 0]], dtype=np.float32)


class OverlayView(QtWidgets.QOpenGLWidget):
    renderFailed = QtCore.pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(350, 300)
        self.image = None
        self.vertices = self.faces = self.normals = self.matrix = None
        self.dirty_image = self.dirty_mesh = False
        self.texture = self.program = 0
        self.buffers = []
        self.texture_shape = None
        self.topology = None
        self.frame_index = None
        self.renderer_name = ''
        self.failed = False
        self.show_mesh = self.show_skeleton = self.show_angles = True
        self.pose_points = None
        self.pose_angles = {}

    def set_layers(self, mesh, skeleton, angles):
        self.show_mesh, self.show_skeleton, self.show_angles = mesh, skeleton, angles
        self.update()

    def set_pose(self, points, angles, valid=None):
        self.pose_valid = valid
        self.pose_points = points
        self.pose_angles = angles
        self.update()

    def draw_pose(self):
        if self.image is None or not self.show_skeleton or self.pose_points is None:return
        # Draw in the exact image viewport established by draw_frame (including DPR).
        points = np.asarray(self.pose_points,dtype=np.float32).copy()
        points[:,0] = points[:,0]*2/self.image.width()-1
        points[:,1] = 1-points[:,1]*2/self.image.height()
        GL.glUseProgram(0)
        GL.glDisable(GL.GL_DEPTH_TEST);GL.glDisable(GL.GL_TEXTURE_2D);GL.glDisable(GL.GL_LIGHTING)
        GL.glDisable(GL.GL_BLEND)
        GL.glMatrixMode(GL.GL_PROJECTION);GL.glLoadIdentity()
        GL.glMatrixMode(GL.GL_MODELVIEW);GL.glLoadIdentity()
        GL.glLineWidth(3*self.devicePixelRatioF())
        GL.glBegin(GL.GL_LINES)
        valid=np.isfinite(points).all(axis=1)
        if getattr(self,"pose_valid",None) is not None:valid &= self.pose_valid
        for (a,b),color in zip(EDGES,COLORS):
            if not (valid[a] and valid[b]):continue
            GL.glColor4f(*map(float,color))
            GL.glVertex2f(*map(float,points[a]));GL.glVertex2f(*map(float,points[b]))
        GL.glEnd()
        GL.glPointSize(6*self.devicePixelRatioF());GL.glColor4f(1,1,1,1)
        GL.glBegin(GL.GL_POINTS)
        for i in np.unique(EDGES):
            if valid[i]:GL.glVertex2f(*map(float,points[i]))
        GL.glEnd()

    def set_frame(self, image, vertices, translation, faces, matrix, frame_index):
        self.image = image.convertToFormat(QtGui.QImage.Format_RGBA8888)
        self.vertices = np.ascontiguousarray(vertices + np.asarray(translation).reshape(1,3), dtype=np.float32)
        self.faces = np.ascontiguousarray(faces, dtype=np.uint32)
        self.normals = None  # Compute only if mesh is actually rendered.
        self.matrix = np.asarray(matrix, dtype=np.float32)
        self.frame_index = frame_index
        self.dirty_image = self.dirty_mesh = True
        self.update()

    def clear_frame(self, image=None):
        self.image = image.convertToFormat(QtGui.QImage.Format_RGBA8888) if image is not None else None
        self.vertices = self.normals = None
        self.dirty_image = image is not None
        self.frame_index = None
        self.pose_points = None
        self.pose_angles = {}
        self.update()

    def initializeGL(self):
        self.texture_shape = self.topology = None
        self.dirty_image = self.image is not None
        self.dirty_mesh = self.vertices is not None
        try:
            self.renderer_name = GL.glGetString(GL.GL_RENDERER).decode(errors='replace')
            # Avoid this environment's PyOpenGL scalar-ctypes texture wrapper.
            texture_ids = np.empty(1, dtype=np.uint32)
            raw_gen_textures(1, texture_ids)
            self.texture = int(texture_ids[0])
            buffer_ids = np.empty(3, dtype=np.uint32)
            GL.glGenBuffers(3, buffer_ids)
            self.buffers = list(map(int,buffer_ids))
            self.program = compileProgram(compileShader('''
                #version 120
                varying vec3 n;
                void main() {
                    n = gl_Normal * vec3(1.,-1.,-1.);
                    gl_Position = gl_ProjectionMatrix * gl_Vertex;
                }''', GL.GL_VERTEX_SHADER), compileShader('''
                #version 120
                varying vec3 n;
                void main() {
                    vec3 v = normalize(n);
                    float key = max(dot(v,normalize(vec3(-0.4,0.6,1.))),0.);
                    float fill = max(dot(v,normalize(vec3(0.8,-0.2,0.6))),0.);
                    gl_FragColor = vec4(vec3(.57,.77,.94)*(.40+.45*key+.15*fill),1.);
                }''', GL.GL_FRAGMENT_SHADER))
            self.context().aboutToBeDestroyed.connect(self.release)
        except Exception as exc:
            traceback.print_exc()
            self.failed = True
            self.renderFailed.emit(str(exc))

    def paintGL(self):
        GL.glClearColor(.043,.067,.125,1.)
        GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)
        if self.failed or self.image is None or not self.texture:
            return
        try:
            self.draw_frame()
            self.draw_pose()
        except Exception as exc:
            traceback.print_exc()
            self.failed = True
            self.renderFailed.emit(str(exc))

    def draw_frame(self):
        width, height = self.image.width(), self.image.height()
        dpr = self.devicePixelRatioF()
        vw, vh = int(self.width()*dpr), int(self.height()*dpr)
        scale = min(vw/width,vh/height)
        rw,rh = round(width*scale),round(height*scale)
        GL.glViewport((vw-rw)//2,(vh-rh)//2,rw,rh)
        GL.glUseProgram(0)
        GL.glDisable(GL.GL_DEPTH_TEST);GL.glDisable(GL.GL_CULL_FACE);GL.glDisable(GL.GL_LIGHTING);GL.glDisable(GL.GL_BLEND)
        GL.glMatrixMode(GL.GL_PROJECTION);GL.glLoadIdentity()
        GL.glMatrixMode(GL.GL_MODELVIEW);GL.glLoadIdentity()
        GL.glEnable(GL.GL_TEXTURE_2D);GL.glBindTexture(GL.GL_TEXTURE_2D,self.texture)
        if self.dirty_image:
            ptr=self.image.constBits();ptr.setsize(self.image.sizeInBytes())
            pixels=np.frombuffer(ptr,dtype=np.uint8).reshape(height,width,4)
            GL.glPixelStorei(GL.GL_UNPACK_ALIGNMENT,1)
            if self.texture_shape != (width,height):
                GL.glTexImage2D(GL.GL_TEXTURE_2D,0,GL.GL_RGBA,width,height,0,GL.GL_RGBA,GL.GL_UNSIGNED_BYTE,pixels)
                GL.glTexParameteri(GL.GL_TEXTURE_2D,GL.GL_TEXTURE_MIN_FILTER,GL.GL_LINEAR)
                GL.glTexParameteri(GL.GL_TEXTURE_2D,GL.GL_TEXTURE_MAG_FILTER,GL.GL_LINEAR)
                self.texture_shape=(width,height)
            else:
                GL.glTexSubImage2D(GL.GL_TEXTURE_2D,0,0,0,width,height,GL.GL_RGBA,GL.GL_UNSIGNED_BYTE,pixels)
            self.dirty_image=False
        GL.glColor4f(1,1,1,1)
        GL.glBegin(GL.GL_QUADS)
        for uv,xy in [((0,1),(-1,-1)),((1,1),(1,-1)),((1,0),(1,1)),((0,0),(-1,1))]:
            GL.glTexCoord2f(*uv);GL.glVertex2f(*xy)
        GL.glEnd();GL.glDisable(GL.GL_TEXTURE_2D)
        if self.vertices is None or not self.show_mesh:
            return
        vbo,nbo,ebo=self.buffers
        if self.dirty_mesh:
            if self.normals is None:self.normals=vertex_normals(self.vertices,self.faces)
            for buffer,data in [(vbo,self.vertices),(nbo,self.normals)]:
                GL.glBindBuffer(GL.GL_ARRAY_BUFFER,buffer)
                GL.glBufferData(GL.GL_ARRAY_BUFFER,data.nbytes,data,GL.GL_STREAM_DRAW)
            if self.topology is None or not np.array_equal(self.faces,self.topology):
                GL.glBindBuffer(GL.GL_ELEMENT_ARRAY_BUFFER,ebo)
                GL.glBufferData(GL.GL_ELEMENT_ARRAY_BUFFER,self.faces.nbytes,self.faces,GL.GL_STATIC_DRAW)
                self.topology=self.faces.copy()
            self.dirty_mesh=False
        GL.glMatrixMode(GL.GL_PROJECTION)
        GL.glLoadMatrixf(np.ascontiguousarray(camera_projection(self.matrix,width,height).T))
        GL.glEnable(GL.GL_DEPTH_TEST);GL.glDepthFunc(GL.GL_LESS);GL.glDepthMask(True)
        GL.glUseProgram(self.program)
        GL.glEnableClientState(GL.GL_VERTEX_ARRAY);GL.glEnableClientState(GL.GL_NORMAL_ARRAY)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER,vbo);GL.glVertexPointer(3,GL.GL_FLOAT,0,ctypes.c_void_p(0))
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER,nbo);GL.glNormalPointer(GL.GL_FLOAT,0,ctypes.c_void_p(0))
        GL.glBindBuffer(GL.GL_ELEMENT_ARRAY_BUFFER,ebo)
        GL.glDrawElements(GL.GL_TRIANGLES,self.faces.size,GL.GL_UNSIGNED_INT,ctypes.c_void_p(0))
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER,0);GL.glBindBuffer(GL.GL_ELEMENT_ARRAY_BUFFER,0)
        GL.glDisableClientState(GL.GL_VERTEX_ARRAY);GL.glDisableClientState(GL.GL_NORMAL_ARRAY)
        GL.glUseProgram(0)

    def release(self):
        if not self.texture and not self.buffers and not self.program:
            return
        self.makeCurrent()
        if self.texture:GL.glDeleteTextures(np.array([self.texture], dtype=np.uint32))
        if self.buffers:GL.glDeleteBuffers(len(self.buffers),np.asarray(self.buffers,dtype=np.uint32))
        if self.program:GL.glDeleteProgram(self.program)
        self.texture=self.program=0;self.buffers=[]
        self.doneCurrent()
