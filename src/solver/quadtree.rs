use nalgebra::Vector2;

#[derive(Clone, Copy)]
pub struct Rect{
    pub(crate) x:f32, //centered at (x, y), w distance from center to edge, h distance from center to edge
    pub(crate) y:f32,
    pub(crate) w:f32,
    pub(crate) h:f32
}

impl Rect {
    pub fn new(x: f32, y: f32, w: f32, h: f32) -> Rect {
        return Rect { x, y, w, h };
    }

    pub fn point_is_in(&self, point: Vector2<f32>) -> bool {
        
        point.x >= self.x - self.w && 
        point.x <= self.x + self.w && 
        point.y >= self.y - self.h && 
        point.y <= self.y + self.h
    }

    pub fn intersects(&self, other: Rect) -> bool {
      
        let x_overlap = (self.x + self.w) >= (other.x - other.w) && 
                       (other.x + other.w) >= (self.x - self.w);
        
        
        let y_overlap = (self.y + self.h) >= (other.y - other.h) && 
                       (other.y + other.h) >= (self.y - self.h);
        
        
        x_overlap && y_overlap
    }

    pub fn intersects_cone(
        &self,
        center: Vector2<f32>,
        _direction: Vector2<f32>, // unused
        radius: f32,
        _angle: f32,              // unused
    ) -> bool {
        // Clamp circle center to the rectangle bounds
        let closest_x = center.x.clamp(self.x - self.w, self.x + self.w);
        let closest_y = center.y.clamp(self.y - self.h, self.y + self.h);

        let dx = center.x - closest_x;
        let dy = center.y - closest_y;

        (dx * dx + dy * dy) <= radius * radius
    }
}

pub struct QuadTree{
    boundary: Rect,
    capacity: i32,
    indices: Vec<i32>,
    points: Vec<Vector2<f32>>,
    northeast: Option<Box<QuadTree>>,
    northwest: Option<Box<QuadTree>>,
    southeast: Option<Box<QuadTree>>,
    southwest: Option<Box<QuadTree>>,

    divided: bool
}

impl QuadTree{
    pub fn new(boundary: Rect, capacity: i32) -> Self {
        QuadTree {
            boundary: boundary,
            capacity: capacity,
            indices: Vec::new(),
            points: Vec::new(),
            northeast: None,
            northwest: None,
            southeast: None,
            southwest: None,
            divided: false
        }
    }

    pub fn insert(&mut self, point: &Vector2<f32>, index: i32) {
       
        if !self.boundary.point_is_in(*point) {
            return;
        }
    
        
        if self.indices.len() < self.capacity as usize {
            self.indices.push(index);
            self.points.push(*point);
        } else {
            
            if !self.divided {
                self.subdivide();
            }
    
            let x = point.x;
            let y = point.y;
            let center_x = self.boundary.x;
            let center_y = self.boundary.y;
    
            if x >= center_x {
                if y >= center_y {
                    self.southeast.as_mut().unwrap().insert(point, index);
                } else {
                    self.northeast.as_mut().unwrap().insert(point, index);
                }
            } else {
                if y >= center_y {
                    self.southwest.as_mut().unwrap().insert(point, index);
                } else {
                    self.northwest.as_mut().unwrap().insert(point, index);
                }
            }
        }
    }

    
    pub fn clear(&mut self) {
        self.indices.clear();
        self.points.clear();
        if self.divided {
            self.northeast.as_mut().unwrap().clear();
            self.northwest.as_mut().unwrap().clear();
            self.southeast.as_mut().unwrap().clear();
            self.southwest.as_mut().unwrap().clear();
        }
        self.divided = false;
    }
    
    
    pub fn subdivide(&mut self){
        self.divided = true;
        let x: f32 = self.boundary.x;
        let y: f32 = self.boundary.y;
        let hh: f32 = self.boundary.h / 2.0;
        let hw: f32 = self.boundary.w / 2.0;

        self.northwest = Some(Box::new(QuadTree::new(Rect::new(x - hw, y - hh, hw, hh), self.capacity)));
        self.northeast = Some(Box::new(QuadTree::new(Rect::new(x + hw, y - hh, hw, hh), self.capacity)));
        self.southwest = Some(Box::new(QuadTree::new(Rect::new(x - hw, y + hh, hw, hh), self.capacity)));
        self.southeast = Some(Box::new(QuadTree::new(Rect::new(x + hw, y + hh, hw, hh), self.capacity)));
    }
    
    pub fn query_cone(
        &self,
        center: Vector2<f32>,
        _direction: Vector2<f32>, // unused
        radius: f32,
        _angle: f32,               // unused
    ) -> Vec<i32> {
        let mut found = Vec::new();

        if radius <= 0.0 {
            return found; // invalid circle
        }

        if !self.boundary.intersects_cone(center, _direction, radius, _angle) {
            return found;
        }

        for i in 0..self.indices.len() {
            if (self.points[i] - center).norm_squared() <= radius * radius {
                found.push(self.indices[i]);
            }
        }

        if self.divided {
            found.extend(self.northwest.as_ref().unwrap().query_cone(center, _direction, radius, _angle));
            found.extend(self.northeast.as_ref().unwrap().query_cone(center, _direction, radius, _angle));
            found.extend(self.southwest.as_ref().unwrap().query_cone(center, _direction, radius, _angle));
            found.extend(self.southeast.as_ref().unwrap().query_cone(center, _direction, radius, _angle));
        }

        found
    }

    pub fn query(&self, rect: &Rect)->Vec<i32>{
        let mut found: Vec<i32> = Vec::new();
        if !self.boundary.intersects(*rect){
            return found;
        }

        for i in 0..self.indices.len(){
            if rect.point_is_in(self.points[i]){
                found.push(self.indices[i]);
            }
        }

        if self.divided {
            found.extend(self.northwest.as_ref().unwrap().query(rect));
            found.extend(self.northeast.as_ref().unwrap().query(rect));
            found.extend(self.southwest.as_ref().unwrap().query(rect));
            found.extend(self.southeast.as_ref().unwrap().query(rect));
        }

        return found;
    }
}