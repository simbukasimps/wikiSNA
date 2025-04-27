// Simple 2D vector class for graph visualization
class Vec2 {
    constructor(x = 0, y = 0) {
        this.x = x;
        this.y = y;
    }
    
    add(other) {
        return new Vec2(this.x + other.x, this.y + other.y);
    }
    
    sub(other) {
        return new Vec2(this.x - other.x, this.y - other.y);
    }
    
    mult(scalar) {
        return new Vec2(this.x * scalar, this.y * scalar);
    }
    
    div(scalar) {
        if (scalar === 0) {
            console.error("Division by zero");
            return new Vec2(0, 0);
        }
        return new Vec2(this.x / scalar, this.y / scalar);
    }
    
    magnitude() {
        return Math.sqrt(this.x * this.x + this.y * this.y);
    }
    
    normalize() {
        const mag = this.magnitude();
        if (mag === 0) {
            return new Vec2(0, 0);
        }
        return this.div(mag);
    }
    
    distance(other) {
        return this.sub(other).magnitude();
    }
    
    static lerp(v1, v2, t) {
        return v1.add(v2.sub(v1).mult(t));
    }
}
