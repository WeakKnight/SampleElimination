class MaxHeap:
    def __init__(self, source):
        self.source = source.copy()
        self.positions = []
        for idx, _ in enumerate(self.source):
            self.positions.append(idx)
        print(self.positions)
        self.quick_sort()
        for pos in self.positions:
            print(self.get_value_by_pos(pos))
        
    def get_value_by_pos(self, pos):
        return self.source[self.positions[pos]]
    
    def is_smaller_then(self, a, b):
        return self.get_value_by_pos(a) < self.get_value_by_pos(b)

    def swap(self, a, b):
        temp = self.positions[a]
        self.positions[a] = self.positions[b]
        self.positions[b] = temp;    
    
    def quick_sort(self):
        for i in range(len(self.positions) - 1):
            for j in range(len(self.positions) - 1 - i):
                if self.is_smaller_then(j, j + 1):
                    self.swap(j, j + 1)

    def move(self):
        print("not implemented")

    def move_up(self):
        print("not implemented")
    
    def move_down(self):
        print("not implemented")