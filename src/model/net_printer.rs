use io::StdoutLock;

use crate::*;
use crate::model::*;
//maybe I should just be using StdoutLock as work in progress?
pub struct NetPrinter<'a> {
    line: String,
    stdoutlock: StdoutLock<'a>,
    ring_num: usize,
    max_length: usize,
    max_window_width: usize,
    last_line_width: usize,
    vert_char: char,
    vert_char_minor: char,
    horz_char: char,
    grad_0: char,
    grad_1: char,
    grad_2: char,
    grad_3: char,
    grad_4: char,
}

//i like this.
//whenever you find a catagory of work, 
//what would I call it if it were a machine?
//It feels overly specific. I couldn't generalize
//this to any other purpose. What would it look
//like if I did and is there a premade solution?
impl<'a> NetPrinter<'a> {
    pub fn new() -> NetPrinter<'a> {
        NetPrinter {
            line: String::new(),
            stdoutlock: io::stdout().lock(),
            ring_num: 0,
            max_length: 230,
            max_window_width: 10,
            last_line_width: 0,
            vert_char:       '║',
            vert_char_minor: '│',
            horz_char:       '─',
            grad_0:          '█',
            grad_1:          '▓',
            grad_2:          '▒',
            grad_3:          '░',
            grad_4:          ' ',
        }
    }
    //I could try to add a BuildState and FreeState.
    //my methods are line_constructors,
    //which puts the printer into a work lose-able state,
    //or line_enders ie carriage_return,
    pub fn display_net(&mut self, model: &Model) {
        for l in 1..model.ring_net.len() {
            for r in 0..model.ring_net[l].len() {
                self.display_ring(&model, l, r);
            }
            self.commit_bar();
        }
        
    }
    fn clear_line(&mut self) {
        self.last_line_width = self.line.chars().count();
        self.line.clear();
    }
    fn commit_line(&mut self) {
        self.truncate_line();
        let _ = writeln!(self.stdoutlock, "{}", self.line);
        self.clear_line();
        let _ = self.stdoutlock.flush();
    }
    fn commit_bar(&mut self) {
        let width = self.last_line_width;
        let ch = self.horz_char;
        for _ in 0..width {
            self.line.push(ch);
        }
        self.commit_line(); 
    }
    
    //I dont know how to reverse the meaning of this name...
    // while myprinter.is_not_overflow() {} ?
    //I still like the idea of an iterator-like function
    //that tells you when full capacity is encountered.
    fn is_overflow(&self) -> bool {
        // String.len() is decidedly constant time,
        // just like all other .len() types. The 
        // trivial understanding. Remember that 'char's
        // must be walked to find the count. The 
        // explicit .chars().count() is intentionally
        // describing the process of the nontrivial algorithm. 
        let length = self.line.chars().count();
        let maximum = self.max_length;
        
        if length > maximum {
            return true
        } else { 
            return false
        }
    }
    fn truncate_line(&mut self) {
        let length = &self.line.chars().count();
        let maximum = &self.max_length;
        if length <= maximum { return }
        let difference = length - maximum;
        for _ in 0..difference {
            let _ = self.line.pop();
        }
        self.line.push_str("...");
    }
    fn push_vert(&mut self) {
        self.line.push(self.vert_char);
    }
    fn push_vert_minor(&mut self) {
        self.line.push(self.vert_char_minor);
    }
    fn push_horz(&mut self) {
        self.line.push(self.horz_char);
    }
    fn push_grad(&mut self, nums: &[f32]) {
        assert!(nums.len() <= self.max_window_width);
        
        for val in nums {
            let val = *val;
            if                val < -0.6 { self.line.push(self.grad_4); continue }
            if val >= -0.6 && val < -0.2 { self.line.push(self.grad_3); continue }
            if val >= -0.2 && val <= 0.2 { self.line.push(self.grad_2); continue }
            if val >   0.2 && val <= 0.6 { self.line.push(self.grad_1); continue }
            if val >   0.6               { self.line.push(self.grad_0); continue }
            panic!()
        }
    }
    fn push_bias(&mut self, bias: &f32, width: usize) {
        
        let val = *bias;
        if                val < -0.6 { 
            for _ in 0..width {
                self.line.push(self.grad_4);
            }
        }
        if val >= -0.6 && val < -0.2 {
            for _ in 0..width {
                self.line.push(self.grad_3);
            }
        }
        if val >= -0.2 && val <= 0.2 {
            for _ in 0..width {
                self.line.push(self.grad_2);
            }
        }
        if val >   0.2 && val <= 0.6 {
            for _ in 0..width {
                self.line.push(self.grad_1);
            }
        }
        if val >   0.6               {
            for _ in 0..width {
                self.line.push(self.grad_0);
            }
        }
        return
    }
    fn push_indent(&mut self) {
        for _ in 0..self.ring_num {
            self.line.push(' ');
            self.line.push(' ');
        }
    }
    fn display_ring(&mut self, model: &Model, l: usize, r: usize) {
        self.line.clear();
        
        self.ring_num = r;
        let weight_count = Model::source_activations_count(&model.ring_net, &model.ring_net[l][r]);
        let max = self.max_window_width;
        let window_width = weight_count.clamp(0, max);
        
        //a process to print one line at a time until it is done.
        let mut remainder = weight_count;
        let mut target = 0;
        loop { //each line
            self.push_indent();
            self.push_vert();
            
            let mut t = target;
            let is_short = remainder < window_width;
            let short = if is_short {window_width - remainder} 
                else { 0 };
            let width = if is_short { remainder } else { window_width };
            for _ in 0..model.ring_net[l][r].neuron_count {
                let nums = &model.weight_net[l][r][t..t+width];
                self.push_grad(nums);
                for _ in 0..short {
                    self.push_horz();
                }
                self.push_vert();
                
                if self.is_overflow() {
                    break
                }
                t += weight_count;
            }
            self.commit_line();
            target += window_width;
            remainder = remainder.saturating_sub(width);
            if remainder == 0 { break }
        }
        
        self.push_indent();
        self.push_vert_minor();
        
        for bias in &model.bias_net[l][r] {
            self.push_bias(&bias, window_width);
            self.push_vert_minor();
            if self.is_overflow() { break }
        }
        self.commit_line();
    }
}

//methods should be the basic construction material of a machine