import numpy as np

# ! DEPRECATED

matrices = [np.array([ [0, 1] ])]

def matrix(h,dim=3):
    global matrices
    m_shapes = [matrices[0].shape]

    while len(matrices) < h:
        m = matrices[-1]
        index = len(matrices) + 1

        growing_dim = 1 if index % dim == 1 else (
            0 if index % dim == 2 else \
            (index - 1) % dim )
        #print(m_shapes[-1], index, growing_dim)

        nms = []
        for i in range(dim):
            try:
                if i % dim == growing_dim:
                    nms.append(m_shapes[-1][i] * 2)
                else:
                    nms.append(m_shapes[-1][i])
            except IndexError:
                if index == dim:
                    nms.append(2)
                else:
                    continue

        x = np.arange(2**index,dtype=int).reshape(tuple(nms))

        '''
        try:
            print(x[tuple([0 for i in range(dim)])])
        except:
            pass

        '''
        print(x, growing_dim)

        elems = 1
        for d in x.shape:
            elems *= d

        print(x.shape, elems)

        m_shapes.append(tuple(nms))

        mn = None

        index = len(matrices)

        if index % 2 == 1:
            mn = np.zeros((m.shape[1],m.shape[1]),int)
            for i in range(m.shape[0]):
                for j in range(m.shape[1]):
                    mn[i*2][j] = m[i][j] << 1
                    mn[i*2 + 1][j] = (m[i][j] << 1) + 1
        else:
            mn = np.zeros((m.shape[0],m.shape[0]*2),int)
            for i in range(m.shape[0]):
                for j in range(m.shape[0]):
                    mn[i][j*2] = m[i][j] << 1
                    mn[i][j*2 + 1] = (m[i][j] << 1) + 1

        matrices.append(mn)
    '''
    for x in matrices:
        print(x[::-1], x.shape)
    '''
matrix(3)
